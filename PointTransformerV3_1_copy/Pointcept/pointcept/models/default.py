import sys
sys.path.append('/data/zuowenhao/PointTransformerV3_1/Pointcept')
import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


def post_process_predictions(coords, pred_logits, sparse_indices=range(2, 10), k_nearest=25):
    """
    对模型预测结果进行后处理：
    1. 获取预测标签
    2. 对8-15标签计算质心
    3. 每个质心最近的25个点保持原预测标签，其余设为标签1
    
    Args:
        coords (torch.Tensor): 点云坐标 [N, 3]
        pred_logits (torch.Tensor): 模型预测logits [N, num_classes]
        sparse_indices (list): 稀疏类别索引，默认为8-15
        k_nearest (int): 每个质心保留最近的点数，默认25
    
    Returns:
        torch.Tensor: 后处理后的logits [N, num_classes]
    """
    device = coords.device
    pred_labels = torch.argmax(pred_logits, dim=1)
    processed_logits = pred_logits.clone()
    
    for class_idx in sparse_indices:
        class_mask = (pred_labels == class_idx)
        if not class_mask.any():
            continue
            
        class_coords = coords[class_mask]
        centroid = class_coords.mean(dim=0, keepdim=True)
        distances = torch.cdist(coords, centroid).squeeze(-1)
        _, nearest_indices = torch.topk(distances, min(k_nearest, len(coords)), largest=False)
        
        reset_mask = class_mask & ~torch.zeros_like(class_mask).scatter_(0, nearest_indices, 1)
        
        # 使用更温和的方式修改 logits，保持梯度
        if reset_mask.any():
            # 创建一个 one-hot 向量，用于平滑地调整 logits
            target_logits = torch.zeros_like(processed_logits[reset_mask])
            target_logits[:, 1] = 10.0  # 类别1的高置信度
            
            # 使用加权方式而不是直接赋值
            processed_logits[reset_mask] = target_logits
        
    return processed_logits




@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        # 新增参数
        enable_prediction_postprocess=True,
        sparse_indices=range(8, 16),
        k_nearest=25,
        # 新增：控制是否仅在推理时应用后处理
        postprocess_inference_only=True,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        
        # 后处理相关参数
        self.enable_prediction_postprocess = enable_prediction_postprocess
        self.sparse_indices = list(sparse_indices)
        self.k_nearest = k_nearest
        self.postprocess_inference_only = postprocess_inference_only
        

    def apply_prediction_postprocess(self, coords, seg_logits, offset=None):
        """
        应用预测后处理
        """
        if not self.enable_prediction_postprocess:
            return seg_logits
            
        # 如果设置为仅推理时应用，且当前在训练模式，则跳过后处理
        if self.postprocess_inference_only and self.training:
            return seg_logits
            
        if offset is not None:
            # 批次处理
            batch_size = len(offset)
            processed_logits_list = []
            
            start_idx = 0
            for i in range(batch_size):
                end_idx = offset[i]
                
                # 获取当前批次的数据
                batch_coords = coords[start_idx:end_idx]
                batch_logits = seg_logits[start_idx:end_idx]
                
                # 应用后处理
                batch_processed = post_process_predictions(
                    batch_coords, batch_logits, self.sparse_indices, self.k_nearest
                )
                processed_logits_list.append(batch_processed)
                
                start_idx = end_idx
            
            return torch.cat(processed_logits_list, dim=0)
        else:
            # 单个点云处理
            return post_process_predictions(
                coords, seg_logits, self.sparse_indices, self.k_nearest
            )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        if isinstance(point, Point):
            feat = point.feat
            coords = point.coord  # 获取点云坐标用于后处理
        else:
            feat = point
            coords = input_dict.get("coord", None)
            
        seg_logits = self.seg_head(feat)
        
        # train - 训练时不应用后处理
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
            
        # eval - 验证时也不应用后处理，确保损失一致性
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
            
        # test - 仅测试时应用后处理
        else:
            if coords is not None:
                offset = input_dict.get("offset", None)
                seg_logits = self.apply_prediction_postprocess(coords, seg_logits, offset)
                
                # 可选：打印后处理统计信息
                if torch.rand(1).item() < 1.0:  # 100%的概率打印
                    try:
                        pred_labels = torch.argmax(seg_logits, dim=1)
                        print("推理后处理统计:")
                        for class_idx in self.sparse_indices:
                            count = (pred_labels == class_idx).sum().item()
                            if count > 0:
                                print(f"  类别 {class_idx}: {count} 个点")
                    except:
                        pass
            
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=208,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
            
        cls_logits = self.cls_head(feat)
        # Training mode: return loss and cls_logits
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)

        # Evaluation mode: return only cls_logits
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)