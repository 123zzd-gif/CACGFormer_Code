"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from .builder import LOSSES


def _hash_grid_coord(grid_coord: torch.Tensor, batch: torch.Tensor = None):
    """
    把 (x,y,z) 的 grid_coord 映射成一个 1D voxel_id，用于 scatter。
    grid_coord: (N, 3) int tensor
    batch: (N,) optional
    """
    # 转 int64 防止溢出
    gc = grid_coord.to(torch.int64)
    x, y, z = gc[:, 0], gc[:, 1], gc[:, 2]

    # 选择足够大的基数避免碰撞（假设你的 grid_coord 范围不会极端大）
    # 也可以改成更稳的 FNV，但这个足够快且可用。
    base1 = 1000003
    base2 = 10007

    voxel_id = x * base1 * base1 + y * base1 + z

    if batch is not None:
        voxel_id = voxel_id + batch.to(torch.int64) * (base1 * base1 * base1 + base2)

    return voxel_id


@LOSSES.register_module()
class NormalVariationBoundaryLoss(nn.Module):
    """
    边界感知损失：用“点法向量 vs 所属体素均值法向量”的差异作为边界强度，
    对高边界强度点赋予更大监督权重。

    - 不做 KNN（避免 O(N^2)）
    - 体素内法向量方差/夹角差能有效指示“几何变化剧烈区域”
    """

    def __init__(
        self,
        loss_weight=1.0,
        ignore_index=-1,
        boundary_scale=3.0,     # 边界增益强度（越大越偏向边界）
        gamma=1.0,              # 对边界强度的幂次（>1 更聚焦极强边界）
        detach_weight=True,     # 是否对权重图 detach（通常建议 True，训练更稳）
        use_cosine=True,        # True: 用 1-cos 作为差异；False: 用 L2 差异
        # ========== 耳甲区专用 ==========
        ear_concha_class_id=None,   # 例如耳甲区对应的类别 id（你数据里 "erjiaqu" 的 index）
        concha_boost=1.5,           # 耳甲区相关边界再额外放大
        concha_only_on_boundary=True,  # True: 只在“耳甲区 vs 非耳甲区”边界点额外放大
        boundary_thresh=0.25,       # concha_only_on_boundary=True 时的边界阈值（0~1）
        # =================================
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.boundary_scale = boundary_scale
        self.gamma = gamma
        self.detach_weight = detach_weight
        self.use_cosine = use_cosine

        self.ear_concha_class_id = ear_concha_class_id
        self.concha_boost = concha_boost
        self.concha_only_on_boundary = concha_only_on_boundary
        self.boundary_thresh = boundary_thresh

    def forward(self, pred, target, grid_coord=None, normal=None, batch=None, **kwargs):
        """
        pred:   (N, C) logits
        target: (N,)   long
        grid_coord: (N,3) int (来自 GridSample return_grid_coord=True + Collect keys)
        normal:    (N,3) float (单位法向量)
        batch:     (N,) optional
        """
        # 基本检查
        if grid_coord is None or normal is None:
            # 没拿到法向量/体素坐标就无法做边界感知：退化为普通 CE
            return F.cross_entropy(pred, target, ignore_index=self.ignore_index) * self.loss_weight

        # mask 掉 ignore
        valid = (target != self.ignore_index)
        if valid.sum() == 0:
            return pred.sum() * 0.0

        pred_v = pred[valid]
        tgt_v = target[valid]
        gc_v = grid_coord[valid]
        n_v = normal[valid]
        b_v = batch[valid] if batch is not None else None

        # 体素 id
        voxel_id = _hash_grid_coord(gc_v, b_v)

        # 体素均值法向量（先 mean 再 normalize）
        voxel_mean_n = torch_scatter.scatter_mean(n_v, voxel_id, dim=0)
        voxel_mean_n = F.normalize(voxel_mean_n, dim=-1, eps=1e-6)

        # 把体素均值回填到点
        mean_n_per_point = voxel_mean_n[torch_scatter.scatter_max(voxel_id, voxel_id, dim=0)[1]][0]  # 不可靠
        # ↑ 上面这种写法不对：scatter_max 返回的是聚合值和 argmax，不是“映射”。
        # 正确做法：用 unique + inverse 映射（更稳，速度也够快）

        uniq, inv = torch.unique(voxel_id, sorted=False, return_inverse=True)
        mean_n_per_point = voxel_mean_n[inv]

        # 法向量差异 => 边界强度
        n_v = F.normalize(n_v, dim=-1, eps=1e-6)
        if self.use_cosine:
            # 1 - cos 越大，夹角变化越大
            cos = (n_v * mean_n_per_point).sum(dim=-1).clamp(-1.0, 1.0)
            diff = 1.0 - cos
        else:
            diff = torch.norm(n_v - mean_n_per_point, dim=-1)

        # 归一化到 [0,1]：避免不同 batch/场景尺度差异
        # 用 min-max + eps（也可改成分位数，更鲁棒）
        d_min = diff.min()
        d_max = diff.max()
        diff_norm = (diff - d_min) / (d_max - d_min + 1e-6)
        diff_norm = diff_norm.clamp(0.0, 1.0)

        # 权重：1 + scale * diff^gamma
        w = 1.0 + self.boundary_scale * (diff_norm ** self.gamma)

        # ========== 耳甲区专用增强 ==========
        if self.ear_concha_class_id is not None:
            concha_mask = (tgt_v == int(self.ear_concha_class_id))

            if self.concha_only_on_boundary:
                boundary_mask = (diff_norm > self.boundary_thresh)
                boost_mask = concha_mask & boundary_mask
            else:
                boost_mask = concha_mask

            if boost_mask.any():
                w = w * torch.where(boost_mask, torch.tensor(self.concha_boost, device=w.device), torch.tensor(1.0, device=w.device))
        # ===================================

        if self.detach_weight:
            w = w.detach()

        # Weighted CE（逐点）
        ce = F.cross_entropy(pred_v, tgt_v, ignore_index=self.ignore_index, reduction="none")
        loss = (ce * w).mean()

        return loss * self.loss_weight

@LOSSES.register_module()
class ClassBalancedRegularizationLoss(nn.Module):
   # """类别平衡正则化损失函数"""
    
    def __init__(self, weight_reg_scale=0.01, loss_weight=1.0, ignore_index=-1):
        super().__init__()
        self.weight_reg_scale = weight_reg_scale
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        
    def forward(self, input, target, model=None):
        #"""
        #Args:
        #    input: 模型预测输出 [N, C]
         #   target: 真实标签 [N]
          #  model: 模型实例，用于获取ClassBalancedAttention模块
        #"""
        if model is None:
            # 如果没有提供模型，只返回0损失
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # 收集所有ClassBalancedAttention模块的正则化损失
        reg_loss = torch.tensor(0.0, device=input.device)
        count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'get_weight_regularization_loss'):
                reg_loss += module.get_weight_regularization_loss()
                count += 1
        
        # 如果有多个模块，取平均
        if count > 0:
            reg_loss = reg_loss / count
        
        return self.loss_weight * self.weight_reg_scale * reg_loss

###之前的
# @LOSSES.register_module()
# class EnhancedMultiLoss(nn.Module):
#    # """增强的多损失组合"""
    
#     def __init__(self, losses_config, model_accessor=None):
#         super().__init__()
#         self.losses_config = losses_config
#         self.model_accessor = model_accessor
        
#         # 构建所有损失函数
#         self.losses = nn.ModuleList()
#         from pointcept.models.builder import build_loss
        
#         for loss_cfg in losses_config:
#             loss_fn = build_loss(loss_cfg)
#             self.losses.append(loss_fn)
    
#     def forward(self, input, target, model=None):
#         #"""计算组合损失"""
#         total_loss = 0
#         loss_dict = {}
        
#         for i, loss_fn in enumerate(self.losses):
#             loss_name = self.losses_config[i].get('type', f'loss_{i}')
            
#             # 检查是否是需要模型参数的损失函数
#             if isinstance(loss_fn, ClassBalancedRegularizationLoss):
#                 loss_value = loss_fn(input, target, model)
#             else:
#                 loss_value = loss_fn(input, target)
            
#             total_loss += loss_value
#             loss_dict[loss_name] = loss_value
        
#         loss_dict['total_loss'] = total_loss
#         return total_loss


@LOSSES.register_module()
class EnhancedMultiLoss(nn.Module):
    def __init__(self, losses_config, model_accessor=None):
        super().__init__()
        self.losses_config = losses_config
        self.model_accessor = model_accessor
        self.losses = nn.ModuleList()
        from pointcept.models.builder import build_loss
        for loss_cfg in losses_config:
            self.losses.append(build_loss(loss_cfg))

    def forward(self, input, target, model=None, **kwargs):
        total_loss = 0
        loss_dict = {}
        for i, loss_fn in enumerate(self.losses):
            loss_name = self.losses_config[i].get("type", f"loss_{i}")

            if isinstance(loss_fn, ClassBalancedRegularizationLoss):
                loss_value = loss_fn(input, target, model)
            else:
                # 关键：把 grid_coord / normal / batch 等信息传下去
                loss_value = loss_fn(input, target, **kwargs)

            total_loss += loss_value
            loss_dict[loss_name] = loss_value

        loss_dict["total_loss"] = total_loss
        return total_loss

        
@LOSSES.register_module()
class DynamicClassAwareLoss(nn.Module):
    def __init__(self, num_classes, base_weight=1.0, sparse_weight=5.0, 
                 sparse_indices=range(8, 16), ignore_index=-1, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.base_weight = base_weight
        self.sparse_weight = sparse_weight
        # 明确指定稀疏类别索引(8-15)
        self.sparse_indices = [idx for idx in sparse_indices if idx < num_classes]
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        
        print(f"初始化动态类别感知损失: 基础权重={base_weight}, 稀疏类别权重={sparse_weight}, 稀疏类别={self.sparse_indices}")
        
    def forward(self, pred, target):
        #"""
        #Args:
            #pred (torch.Tensor): 预测结果，形状为 [N, C] 或 [N, C, d1, d2, ...]
            #target (torch.Tensor): 目标标签，形状为 [N] 或 [N, d1, d2, ...]
        #"""
        # 处理预测形状
        if pred.dim() > 2:
            pred = pred.reshape(pred.size(0), pred.size(1), -1)  # (N, C, d1*d2*...)
            pred = pred.transpose(1, 2).reshape(-1, pred.size(1))  # (N*d1*d2..., C)
            
        if target.dim() > 1:
            target = target.reshape(-1)  # (N*d1*d2...)
            
        # 创建权重张量
        weights = torch.ones(self.num_classes, device=pred.device) * self.base_weight
        for idx in self.sparse_indices:
            weights[idx] = self.sparse_weight
        
        # 计算每个类的点数
        valid_mask = target != self.ignore_index
        class_counts = torch.bincount(target[valid_mask], minlength=self.num_classes)
        total_count = class_counts.sum()
        
        # 安全打印类别统计信息
        if self.training and torch.rand(1).item() < 0.01:
            try:
                is_main_process = True
                if torch.distributed.is_initialized():
                    is_main_process = torch.distributed.get_rank() == 0
                
                if is_main_process:
                    print("类别统计(8-15):")
                    for cls_id in self.sparse_indices:
                        print(f"类别 {cls_id}: {class_counts[cls_id].item()} 点")
            except:
                pass
        
        # 基于类频率动态调整权重
        class_freq = class_counts.float() / max(total_count, 1)
        inverse_freq = 1.0 / (class_freq + 1e-10)
        normalized_inverse_freq = inverse_freq / inverse_freq.sum() * self.num_classes
        
        # 最终权重 = 固定权重 * 动态频率权重
        final_weights = weights * normalized_inverse_freq
        
        # 使用交叉熵损失
        loss_fn = nn.CrossEntropyLoss(weight=final_weights, ignore_index=self.ignore_index)
        loss = loss_fn(pred, target)
        
        return loss * self.loss_weight

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
        
        
@LOSSES.register_module()
class IoULoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1):
        super(IoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):

        pred = F.softmax(pred, dim=1)
        batch, C, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)
        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]
        target_one_hot = F.one_hot(target, num_classes=C).float()
        intersection = torch.sum(pred * target_one_hot, dim=0)
        union = torch.sum(pred + target_one_hot, dim=0) - intersection
        iou = intersection / (union + 1e-10)
        
        loss = 1 - iou
        return self.loss_weight * loss.mean()