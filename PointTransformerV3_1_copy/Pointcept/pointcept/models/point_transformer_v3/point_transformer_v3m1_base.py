# -*- coding: utf-8 -*-

"""
Enhanced Point Transformer - V3 Model with Feature Visualization

Based on the original work by Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Enhanced for better small region segmentation and feature visualization
"""
import matplotlib.colors as mcolors

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import time

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


# 从原始模型继承的组件
class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
            
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        else:
            self.act = None

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


# 改进的类别平衡注意力模块，所有权重保存到一个文件
class ClassBalancedAttention(PointModule):
    def __init__(self, channels, num_classes, sparse_indices=range(8, 16), 
                 balance_weights=None, learnable_weights=True, 
                 weight_init_scale=3.0, weight_regularization=0.01,
                 stage_name="unknown", visualization_dir=None, save_weights_every=4):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.sparse_indices = [i for i in sparse_indices if i < num_classes]
        self.learnable_weights = learnable_weights
        self.weight_regularization = weight_regularization
        self.stage_name = stage_name
        
        # 权重保存相关参数
        self.visualization_dir = visualization_dir
        self.save_weights_every = save_weights_every
        self.local_step_counter = 0
        
        # 创建单一权重文件路径
        if self.visualization_dir:
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.weights_file_path = os.path.join(self.visualization_dir, "weights.txt")
            self._create_weights_file()
        
        print(f"初始化类别平衡注意力模块[{stage_name}]: 输入通道={channels}, 类别数={num_classes}, "
              f"稀疏类别={self.sparse_indices}, 可学习权重={learnable_weights}, "
              f"每{save_weights_every}步保存权重到: {self.weights_file_path if self.visualization_dir else 'None'}")
        
        # 初始化权重
        if learnable_weights:
            init_weights = torch.ones(num_classes)
            for idx in self.sparse_indices:
                init_weights[idx] = weight_init_scale
            self.balance_weights = nn.Parameter(init_weights)
            self.weight_activation = nn.Softplus()
        else:
            if balance_weights is None:
                self.balance_weights = torch.ones(num_classes)
                for idx in self.sparse_indices:
                    self.balance_weights[idx] = weight_init_scale
            else:
                self.balance_weights = balance_weights
            self.register_buffer('balance_weights_buffer', self.balance_weights)
        
        # 初始化注意力网络
        self._build_attention_network(channels)
        
        # 保存初始权重
        if self.visualization_dir:
            self._save_weights_to_file(0, is_initial=True)
        
    def _create_weights_file(self):
        """创建权重文件并写入头部信息"""
        with open(self.weights_file_path, 'w', encoding='utf-8') as f:
            f.write("# 类别平衡注意力权重记录\n")
            f.write(f"# 模型阶段: {self.stage_name}\n")
            f.write(f"# 类别数量: {self.num_classes}\n")
            f.write(f"# 稀疏类别索引: {self.sparse_indices}\n")
            f.write(f"# 保存频率: 每{self.save_weights_every}步\n")
            f.write(f"# 创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# ==========================================\n")
            f.write("# 格式: 全局步数 | 本地步数 | 时间 | 类别0权重 | 类别1权重 | ... | 类别15权重 | 稀疏类均值 | 非稀疏类均值\n")
            f.write("# ==========================================\n\n")
        
    def _build_attention_network(self, channels):
        """构建注意力网络"""
        hidden_dim = max(channels // 4, 64)
        self.attention = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes)
        )
        
    def get_balance_weights(self):
        """获取当前的平衡权重"""
        if self.learnable_weights:
            return self.weight_activation(self.balance_weights)
        else:
            return self.balance_weights_buffer
    
    def _save_weights_to_file(self, global_step, is_initial=False):
        """将权重保存到单一文件中"""
        if not self.visualization_dir:
            return
            
        weights = self.get_balance_weights()
        weights_np = weights.detach().cpu().numpy()
        
        # 计算统计信息
        if self.sparse_indices:
            sparse_weights = weights_np[self.sparse_indices]
            sparse_mean = np.mean(sparse_weights)
            
            non_sparse_indices = [i for i in range(len(weights_np)) if i not in self.sparse_indices]
            non_sparse_mean = np.mean(weights_np[non_sparse_indices]) if non_sparse_indices else 0.0
        else:
            sparse_mean = 0.0
            non_sparse_mean = np.mean(weights_np)
        
        # 追加写入文件
        with open(self.weights_file_path, 'a', encoding='utf-8') as f:
            if is_initial:
                f.write(f"# 初始权重 (步数 0)\n")
            
            # 写入数据行: 全局步数 | 本地步数 | 时间 | 各类别权重 | 统计信息
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            weights_str = " | ".join([f"{w:.8f}" for w in weights_np])
            
            f.write(f"{global_step} | {self.local_step_counter} | {timestamp} | {weights_str} | {sparse_mean:.8f} | {non_sparse_mean:.8f}\n")
        
        if is_initial:
            print(f"初始权重已保存到: {self.weights_file_path}")
        else:
            print(f"[步骤 {global_step}] 权重已保存到: {self.weights_file_path}")
    
    def get_weight_regularization_loss(self):
        """计算权重正则化损失"""
        if not self.learnable_weights:
            return torch.tensor(0.0, device=self.balance_weights.device)
        
        l2_loss = torch.sum(self.balance_weights ** 2)
        
        sparse_weights = self.balance_weights[self.sparse_indices]
        common_weights = torch.cat([
            self.balance_weights[:min(self.sparse_indices)],
            self.balance_weights[max(self.sparse_indices)+1:]
        ]) if self.sparse_indices else self.balance_weights
        
        if len(common_weights) > 0:
            sparse_preference_loss = torch.relu(
                common_weights.mean() - sparse_weights.mean() + 1.0
            )
        else:
            sparse_preference_loss = torch.tensor(0.0, device=self.balance_weights.device)
        
        return self.weight_regularization * (l2_loss + sparse_preference_loss)
        
    def forward(self, point, global_step=None):
        """前向传播"""
        self.local_step_counter += 1
        
        # 特征维度检查和调整
        feat_dim = point.feat.shape[1]
        if feat_dim != self.channels:
            print(f"自动调整[{self.stage_name}]: 特征维度 {feat_dim} 与预期通道数 {self.channels} 不匹配")
            device = point.feat.device
            self.channels = feat_dim
            
            hidden_dim = max(feat_dim // 4, 64)
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim).to(device),
                nn.LayerNorm(hidden_dim).to(device),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_classes).to(device)
            )
            
        # 计算注意力权重
        attn_logits = self.attention(point.feat)
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 获取平衡权重
        balance_weights = self.get_balance_weights()
        
        # 检查是否需要保存权重
        if (global_step is not None and 
            global_step > 0 and 
            global_step % self.save_weights_every == 0 and
            self.visualization_dir):
            self._save_weights_to_file(global_step)
        
        # 应用权重调整
        weighted_attn = attn_weights * balance_weights.to(attn_weights.device)
        point.feat = point.feat * (1 + weighted_attn.sum(dim=1, keepdim=True))
        
        return point, attn_weights


class SimpleGlobalEncoder(PointModule):
    """简化版的全局特征编码器"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        
    def forward(self, point):
        point.feat = self.global_encoder(point.feat)
        return point


class FeatureVisualizer:
    """特征可视化工具类"""
    def __init__(self, save_dir="./feature_visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义16种离散颜色，确保视觉区分度
        self.discrete_colors = [
            '#FF0000',  # 红色
            '#00FF00',  # 绿色  
            '#0000FF',  # 蓝色
            '#FFFF00',  # 黄色
            '#FF00FF',  # 洋红
            '#00FFFF',  # 青色
            '#FFA500',  # 橙色
            '#800080',  # 紫色
            '#FFC0CB',  # 粉色
            '#A52A2A',  # 棕色
            '#808080',  # 灰色
            '#000000',  # 黑色
            '#90EE90',  # 浅绿色
            '#FFB6C1',  # 浅粉色
            '#20B2AA',  # 浅海绿色
            '#DDA0DD'   # 梅花色
        ]
        
        # 创建离散颜色映射
        self.discrete_cmap = mcolors.ListedColormap(self.discrete_colors)
        
    def visualize_features(self, features, coords=None, labels=None, 
                        step_name="unknown", sample_size=29800, method="tsne"):
        """可视化高维特征"""
        # 转换为numpy
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
            
        if coords is not None and isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords
            
        if labels is not None and isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
        
        # 差异化采样：标签0-7限制为200个点，标签8-15保持原数量
        if labels_np is not None:
            selected_indices = []
            n_total_points = features_np.shape[0]
            
            # 处理数组长度不一致的情况
            min_length = n_total_points
            if coords_np is not None:
                min_length = min(min_length, coords_np.shape[0])
            min_length = min(min_length, len(labels_np))
            
            print(f"数组长度检查: features={n_total_points}, labels={len(labels_np)}, coords={coords_np.shape[0] if coords_np is not None else 'None'}")
            print(f"使用最小长度: {min_length}")
            
            # 截取到相同长度
            features_np = features_np[:min_length]
            labels_np = labels_np[:min_length]
            if coords_np is not None:
                coords_np = coords_np[:min_length]
            
            n_total_points = min_length
            
            for label in np.unique(labels_np):
                label_indices = np.where(labels_np == label)[0]
                
                # 确保索引不超出边界
                valid_indices = label_indices[label_indices < n_total_points]
                if len(valid_indices) != len(label_indices):
                    print(f"Warning: Found {len(label_indices) - len(valid_indices)} invalid indices for label {label}")
                
                if label <= 7:  # 标签0-7，最多保留200个点
                    n_samples = min(200, len(valid_indices))
                else:  # 标签8-15，保持原数量
                    n_samples = len(valid_indices)
                
                if n_samples < len(valid_indices):
                    # 随机采样
                    sampled_indices = np.random.choice(valid_indices, n_samples, replace=False)
                else:
                    sampled_indices = valid_indices
                
                selected_indices.extend(sampled_indices.tolist())
            
            # 转换为numpy数组，确保索引有效，并打乱顺序
            selected_indices = np.array(selected_indices)
            selected_indices = selected_indices[selected_indices < n_total_points]  # 再次确保索引有效
            selected_indices = np.unique(selected_indices)  # 去除重复索引
            np.random.shuffle(selected_indices)
            
            print(f"原始点数: {n_total_points}, 采样后点数: {len(selected_indices)}")
            
            # 应用选择的索引
            features_np = features_np[selected_indices]
            if coords_np is not None:
                coords_np = coords_np[selected_indices]
            labels_np = labels_np[selected_indices]
            
            print(f"采样后点云分布:")
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            for label, count in zip(unique_labels, counts):
                status = "(limited to 200)" if label <= 7 else "(full)"
                print(f"  标签 {label}: {count} 个点 {status}")
        
        else:
            # 如果没有标签，按原来的方式采样
            n_points = features_np.shape[0]
            if n_points > sample_size:
                indices = np.random.choice(n_points, sample_size, replace=False)
                features_np = features_np[indices]
                if coords_np is not None:
                    coords_np = coords_np[indices]
        
        # 降维
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        features_2d = reducer.fit_transform(features_np)
        
        # 创建图形
        fig, axes = plt.subplots(1, 3 if coords_np is not None else 2, figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # 1. 基础特征分布图 - 使用离散颜色
        ax = axes[0]
        if labels_np is not None:
            # 确保标签在0-15范围内
            labels_normalized = labels_np % 16
            
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels_normalized, 
                            cmap=self.discrete_cmap, 
                            s=1, alpha=0.7,
                            vmin=0, vmax=15)  # 设置颜色范围
            
            # 创建离散的颜色条
            cbar = plt.colorbar(scatter, ax=ax, ticks=range(16))
            cbar.set_label('Class Label')
            cbar.set_ticklabels([f'Class {i}' for i in range(16)])
            
            ax.set_title(f'{step_name} - Feature Distribution (Classes 0-7: ≤200pts, 8-15: full)')
        else:
            ax.scatter(features_2d[:, 0], features_2d[:, 1], s=1, alpha=0.6)
            ax.set_title(f'{step_name} - Feature Distribution')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        
        # 2. 特征统计图
        ax = axes[1]
        feature_stats = {
            'mean': np.mean(features_np, axis=0),
            'std': np.std(features_np, axis=0),
            'min': np.min(features_np, axis=0),
            'max': np.max(features_np, axis=0)
        }
        
        channels = range(features_np.shape[1])
        ax.plot(channels, feature_stats['mean'], label='Mean', alpha=0.8)
        ax.fill_between(channels, 
                    feature_stats['mean'] - feature_stats['std'],
                    feature_stats['mean'] + feature_stats['std'], 
                    alpha=0.3, label='±1 Std')
        ax.set_title(f'{step_name} - Feature Statistics')
        ax.set_xlabel('Feature Channel')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 空间分布 - 也使用离散颜色
        if coords_np is not None and len(axes) > 2:
            ax = axes[2]
            if features_np.shape[1] > 0:
                if labels_np is not None:
                    # 使用标签颜色显示空间分布
                    labels_normalized = labels_np % 16
                    scatter = ax.scatter(coords_np[:, 0], coords_np[:, 1], 
                                    c=labels_normalized, 
                                    cmap=self.discrete_cmap, 
                                    s=1, alpha=0.7,
                                    vmin=0, vmax=15)
                    ax.set_title(f'{step_name} - Spatial Distribution (Colored by Class)')
                else:
                    # 使用第一主成分颜色显示
                    first_pc = PCA(n_components=1).fit_transform(features_np).flatten()
                    scatter = ax.scatter(coords_np[:, 0], coords_np[:, 1], 
                                    c=first_pc, cmap='viridis', s=1, alpha=0.6)
                    ax.set_title(f'{step_name} - Spatial Distribution (Colored by 1st PC)')
                
                # 修复: 使用局部变量检查而不是全局检查
                if 'scatter' in locals():
                    plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(coords_np[:, 0], coords_np[:, 1], s=1, alpha=0.6)
                ax.set_title(f'{step_name} - Spatial Distribution')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f"{step_name}_{method}_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存特征统计信息和颜色信息
        stats_path = os.path.join(self.save_dir, f"{step_name}_feature_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Feature Statistics for {step_name}:\n")
            f.write(f"Shape: {features_np.shape}\n")
            f.write(f"Mean: {np.mean(features_np):.6f}\n")
            f.write(f"Std: {np.std(features_np):.6f}\n")
            f.write(f"Min: {np.min(features_np):.6f}\n")
            f.write(f"Max: {np.max(features_np):.6f}\n")
            f.write(f"Per-channel mean: {np.mean(features_np, axis=0)}\n")
            f.write(f"Per-channel std: {np.std(features_np, axis=0)}\n")
            
            # 添加标签分布信息
            if labels_np is not None:
                unique_labels, counts = np.unique(labels_np, return_counts=True)
                f.write(f"\nLabel Distribution (after differential sampling):\n")
                for label, count in zip(unique_labels, counts):
                    status = "(limited to ≤200)" if label <= 7 else "(full sampling)"
                    f.write(f"  Class {label}: {count} points ({count/len(labels_np)*100:.2f}%) {status}\n")
                
                # 添加颜色映射信息
                f.write(f"\nColor Mapping (16 discrete colors):\n")
                for i, color in enumerate(self.discrete_colors):
                    f.write(f"  Class {i}: {color}\n")
                
                # 添加采样策略说明
                f.write(f"\nSampling Strategy:\n")
                f.write(f"  - Classes 0-7: Limited to maximum 200 points each\n")
                f.write(f"  - Classes 8-15: Full sampling (no limit)\n")
        
        print(f"Feature visualization saved to {save_path}")
        print(f"Feature statistics saved to {stats_path}")
        
    def visualize_attention_weights(self, attention_weights, step_name="attention"):
        """可视化注意力权重分布"""
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 注意力权重热力图
        ax = axes[0]
        im = ax.imshow(attention_weights.T, aspect='auto', cmap='viridis')
        ax.set_title(f'{step_name} - Attention Weights Heatmap')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Class Index')
        plt.colorbar(im, ax=ax)
        
        # 2. 每个类别的平均注意力权重 - 使用离散颜色
        ax = axes[1]
        mean_attention = np.mean(attention_weights, axis=0)
        
        # 为每个类别分配对应的离散颜色
        bar_colors = [self.discrete_colors[i % 16] for i in range(len(mean_attention))]
        
        bars = ax.bar(range(len(mean_attention)), mean_attention, color=bar_colors, alpha=0.8)
        ax.set_title(f'{step_name} - Average Attention per Class')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Average Attention Weight')
        
        # 添加类别标签
        ax.set_xticks(range(len(mean_attention)))
        ax.set_xticklabels([f'C{i}' for i in range(len(mean_attention))])
        
        # 高亮稀疏类别（8-15）
        sparse_indices = range(8, 16)
        for i, bar in enumerate(bars):
            if i in sparse_indices and i < len(bars):  # 添加边界检查
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
        
        # 添加图例说明稀疏类别
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='white', edgecolor='red', linewidth=2, label='Sparse Classes (8-15)')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{step_name}_attention_weights.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Attention weights visualization saved to {save_path}")


# 增强版Point Transformer V3模型
@MODELS.register_module("PT-v3m1")
class PointTransformerV3Enhanced(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        # 新增参数
        num_classes=16,
        use_class_balanced_attn=True,
        use_region_aware_pooling=True,
        sparse_indices=range(8, 16),
        class_weights=None,
        # 可视化相关参数
        enable_visualization=True,
        visualization_dir="exp/default_ear_xi__keshihua_29800points_Dloss_weight_2.5_qvbie_test_val/keshihua",
        visualize_every_n_steps=2,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.num_classes = num_classes
        
        # 新增参数
        self.use_class_balanced_attn = use_class_balanced_attn
        self.use_region_aware_pooling = use_region_aware_pooling
        self.sparse_indices = [idx for idx in sparse_indices if idx < num_classes]
        
        # 可视化相关参数
        self.enable_visualization = enable_visualization
        self.visualize_every_n_steps = visualize_every_n_steps
        self.global_step = 0
        
        if self.enable_visualization:
            self.visualizer = FeatureVisualizer(visualization_dir)
            print(f"可视化已启用，每 {visualize_every_n_steps} 步可视化一次，保存路径: {visualization_dir}")

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            
            # 在解码器添加类别平衡注意力模块
            if self.use_class_balanced_attn:
                self.class_attn_modules = nn.ModuleList()
                for s in range(self.num_stages - 1):
                    stage_name = f"decoder_stage_{s}"
                    self.class_attn_modules.append(
                        ClassBalancedAttention(
                            channels=dec_channels[s],
                            num_classes=self.num_classes,
                            sparse_indices=self.sparse_indices,
                            balance_weights=class_weights,
                            stage_name=stage_name,
                            visualization_dir=visualization_dir if enable_visualization else None,
                            save_weights_every=400
                        )
                    )
            
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
                
            # 添加SimpleGlobalEncoder
            if self.use_region_aware_pooling:
                self.simple_global_encoder = SimpleGlobalEncoder(
                    in_channels=enc_channels[0] + sum(dec_channels[:-1]),
                    out_channels=enc_channels[0] + sum(dec_channels[:-1])
                )
    
    def should_visualize(self):
        """判断是否应该进行可视化"""
        return (self.enable_visualization and 
                self.training and 
                self.global_step % self.visualize_every_n_steps == 0)

    def forward(self, data_dict):
        # 增加全局步数计数
        self.global_step += 1
        
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # 原始点数
        original_num_points = point.feat.shape[0]
        
        # 获取坐标信息用于可视化
        coords = point.coord if hasattr(point, 'coord') else None
        labels = data_dict.get('segment', None)
        
        # 1. 嵌入和编码
        point = self.embedding(point)
        embed_feat = point.feat
        
        # 可视化嵌入特征
        if self.should_visualize():
            print(f"[步骤 {self.global_step}] 开始可视化嵌入特征...")
            self.visualizer.visualize_features(
                embed_feat, coords, labels, 
                step_name=f"step_{self.global_step}_embedding"
            )
        
        point = self.enc(point)

        if not self.cls_mode:
            dec_features = [embed_feat]
            pooling_inverses = [None]
            attention_weights_list = []
            
            # 2. 解码并应用类别平衡注意力
            for i in range(len(self.dec)):
                point = self.dec[i](point)
                    
                # 应用类别平衡注意力
                if self.use_class_balanced_attn:
                    rev_idx = len(self.class_attn_modules) - 1 - i
                    if 0 <= rev_idx < len(self.class_attn_modules):
                        # 在应用注意力前保存特征用于可视化
                        pre_attn_feat = point.feat.clone()
                        
                        # 应用注意力并传递全局步数
                        attn_module = self.class_attn_modules[rev_idx]
                        point, attn_weights = attn_module(point, global_step=self.global_step)
                        attention_weights_list.append(attn_weights)
                        
                        # 可视化注意力前后的特征对比
                        if self.should_visualize():
                            stage_name = f"step_{self.global_step}_decoder_stage_{rev_idx}"
                            
                            print(f"[步骤 {self.global_step}] 可视化解码器阶段 {rev_idx}...")
                            
                            # 可视化注意力前的特征
                            self.visualizer.visualize_features(
                                pre_attn_feat, coords, labels,
                                step_name=f"{stage_name}_pre_attention"
                            )
                            
                            # 可视化注意力后的特征
                            self.visualizer.visualize_features(
                                point.feat, coords, labels,
                                step_name=f"{stage_name}_post_attention"
                            )
                            
                            # 可视化注意力权重
                            self.visualizer.visualize_attention_weights(
                                attn_weights, step_name=stage_name
                            )
                
                dec_features.append(point.feat)
                if i < len(self.dec) - 1:
                    if "pooling_inverse" in point.keys():
                        pooling_inverses.append(point["pooling_inverse"])
                    else:
                        pooling_inverses.append(None)
                else:
                    pooling_inverses.append(None)
            
            # 3. 对齐特征
            aligned_features = []
            for idx, (feat, inverse) in enumerate(zip(dec_features, pooling_inverses)):
                if inverse is None:
                    aligned_feat = torch.zeros(
                        original_num_points, feat.shape[1], device=feat.device
                    )
                    if feat.shape[0] == original_num_points:
                        aligned_feat = feat
                    aligned_features.append(aligned_feat)
                else:
                    if inverse.shape[0] != feat.shape[0]:
                        min_size = min(inverse.shape[0], feat.shape[0])
                        inverse = inverse[:min_size]
                        feat = feat[:min_size]
                    aligned_feat = torch.zeros(
                        original_num_points, feat.shape[1], device=feat.device
                    )
                    valid_mask = inverse < original_num_points
                    inverse = inverse[valid_mask]
                    feat = feat[:inverse.shape[0]]
                    aligned_feat.index_add_(0, inverse, feat)
                    aligned_features.append(aligned_feat)
            
            # 4. 拼接特征
            concat_feat = torch.cat(aligned_features, dim=-1)
            point.feat = concat_feat
            
            # 可视化拼接后的特征
            if self.should_visualize():
                print(f"[步骤 {self.global_step}] 可视化拼接特征...")
                self.visualizer.visualize_features(
                    concat_feat, coords, labels,
                    step_name=f"step_{self.global_step}_concat_features"
                )
                
                # 保存特征到文件以供进一步分析
                feature_save_path = os.path.join(
                    self.visualizer.save_dir, 
                    f"step_{self.global_step}_concat_features.pt"
                )
                torch.save({
                    'features': concat_feat,
                    'coords': coords,
                    'labels': labels,
                    'attention_weights': attention_weights_list,
                    'global_step': self.global_step
                }, feature_save_path)
                print(f"特征数据保存到: {feature_save_path}")
            
            # 5. 应用全局特征编码
            if self.use_region_aware_pooling and hasattr(self, 'simple_global_encoder'):
                point = self.simple_global_encoder(point)
                
                # 可视化最终特征
                if self.should_visualize():
                    print(f"[步骤 {self.global_step}] 可视化最终特征...")
                    self.visualizer.visualize_features(
                        point.feat, coords, labels,
                        step_name=f"step_{self.global_step}_final_features"
                    )
                    print(f"[步骤 {self.global_step}] 可视化完成！")
            
        return point
    
    def get_regularization_loss(self):
        """获取所有ClassBalancedAttention模块的正则化损失"""
        reg_loss = torch.tensor(0.0)
        count = 0
        
        for module in self.modules():
            if hasattr(module, 'get_weight_regularization_loss'):
                if reg_loss.device != module.get_weight_regularization_loss().device:
                    reg_loss = reg_loss.to(module.get_weight_regularization_loss().device)
                reg_loss += module.get_weight_regularization_loss()
                count += 1
        
        return reg_loss / max(count, 1)
    
    def get_visualization_stats(self):
        """获取可视化统计信息"""
        return {
            'global_step': self.global_step,
            'visualize_every_n_steps': self.visualize_every_n_steps,
            'enable_visualization': self.enable_visualization,
            'next_visualization_step': (self.global_step // self.visualize_every_n_steps + 1) * self.visualize_every_n_steps,
            'visualization_dir': self.visualizer.save_dir if self.enable_visualization else None
        }
    
    def set_visualization_interval(self, new_interval):
        """动态调整可视化间隔"""
        self.visualize_every_n_steps = new_interval
        print(f"可视化间隔已调整为每 {new_interval} 步")
    
    def reset_global_step(self):
        """重置全局步数计数器"""
        self.global_step = 0
        print("全局步数计数器已重置")
    
    def get_all_balance_weights(self):
        """获取所有ClassBalancedAttention模块的当前权重"""
        weights_dict = {}
        if self.use_class_balanced_attn and hasattr(self, 'class_attn_modules'):
            for i, module in enumerate(self.class_attn_modules):
                weights = module.get_balance_weights()
                weights_dict[f"stage_{i}"] = weights.detach().cpu().numpy()
        return weights_dict
    
    def manual_save_weights(self, step_name="manual"):
        """手动触发权重保存"""
        if self.use_class_balanced_attn and hasattr(self, 'class_attn_modules'):
            for i, module in enumerate(self.class_attn_modules):
                module._save_weights_to_file(f"{step_name}_{i}")
            print(f"手动权重保存完成: {step_name}")

