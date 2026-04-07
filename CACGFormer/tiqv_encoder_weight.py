import torch
import sys
sys.path.append("/home/m/process/PointTransformerV3_1/Pointcept")  # 确保能 import pointcept

path = "/home/m/process/PointTransformerV3_1/Pointcept/tools/exp/default_ear_xi__keshihua_2fenlei_normals_500guanfang_3jihetezheng+loss_zhuyili/model/model_best.pth"
save_path = "/home/m/process/PointTransformerV3_1/Pointcept/tools/exp/default_ear_xi__keshihua_2fenlei_normals_500guanfang_3jihetezheng+loss_zhuyili/model/encoder_pretrained.pth"

ckpt = torch.load(path, map_location="cpu")

# 获取 state_dict
if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
else:
    sd = ckpt

encoder_prefixes = (
    "backbone.embedding.",
    "backbone.enc.",
    "backbone.bio_geo_extractor.",
)

encoder_sd = {k: v for k, v in sd.items() if k.startswith(encoder_prefixes)}

print(f"encoder 参数已保存到: {save_path}, 共 {len(encoder_sd)} 个参数")
torch.save(encoder_sd, save_path)
