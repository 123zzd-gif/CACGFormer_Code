_base_ = ["../_base_/default_runtime.py"]

save_path = "exp/ear_concha_unsup_cluster_ptv3_contrastive"

# =====================================================
# runtime
# =====================================================
batch_size = 1
num_worker = 8
mix_prob = 0.0
empty_cache = True
enable_amp = True

epoch = 200
eval_epoch = 20

# 你已有的“仅编码器预训练权重”
pretrained_encoder_ckpt = "/path/to/your/pretrained_encoder_checkpoint.pth"

# =====================================================
# model
# =====================================================
model = dict(
    type="PT-v3-cluster-contrastive",

    # ---------- 预训练编码器 ----------
    pretrained_encoder_ckpt=pretrained_encoder_ckpt,

    # ---------- 输入 ----------
    in_channels=6,          # coord(3) + normal(3)
    bio_geo_k=20,

    # ---------- 编码器结构 ----------
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),

    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(48, 48, 48, 48, 48),

    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,

    enable_rpe=True,
    enable_flash=False,
    upcast_attention=True,
    upcast_softmax=True,

    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),

    # ---------- 几何增强注意力 ----------
    use_geo_attention=True,
    geo_dim=3,
    geo_hidden_dim=16,
    use_geo_qkv_gate=True,
    use_geo_attn_bias=True,
    gate_residual_scale=0.1,
    pair_geo_hidden_dim=32,
    geo_eps=1e-6,

    # ---------- 聚类 / 对比学习 ----------
    global_dim=512,
    proj_hidden_dim=512,
    proj_dim=128,

    # 聚类簇数：比如胖/瘦/小 -> 3
    num_prototypes=3,
    proto_temperature=0.1,
    contrast_temperature=0.2,

    # loss 权重
    lambda_instance=1.0,
    lambda_hierarchy=0.3,
    lambda_anatomy=0.2,
    lambda_proto=0.1,
    hierarchy_margin=0.05,

    # ---------- 解剖学强弱增强 ----------
    weak_rotate_deg=8.0,
    strong_rotate_deg=20.0,
    weak_translate=0.01,
    strong_translate=0.03,
    weak_drop_ratio=0.05,
    strong_drop_ratio=0.15,
    noise_std_weak=0.002,
    noise_std_strong=0.006,
    protect_ratio=0.25,
)

# =====================================================
# optimizer
# =====================================================
optimizer = dict(
    type="AdamW",
    lr=3e-4,
    weight_decay=0.05,
)

# 不同模块用不同学习率：
# encoder 小 lr；几何模块中 lr；新头大 lr
param_dicts = [
    dict(keyword="encoder", lr=3e-4),
    dict(keyword="bio_geo_extractor", lr=1e-3),
    dict(keyword="geo_q_gate", lr=1e-3),
    dict(keyword="geo_k_gate", lr=1e-3),
    dict(keyword="geo_v_gate", lr=1e-3),
    dict(keyword="pair_geo_mlp", lr=1e-3),
    dict(keyword="pair_geo_norm", lr=1e-3),
    dict(keyword="global_pool", lr=3e-3),
    dict(keyword="projection_head", lr=3e-3),
    dict(keyword="prototype_head", lr=3e-3),
]

scheduler = dict(
    type="CosineAnnealingLR",
    T_max=epoch,
    eta_min=1e-6,
)

# =====================================================
# dataset
# =====================================================
dataset_type = "S3DISDataset"
data_root = "/home/m/process/PointTransformerV3_1/dataset/erjiaqv/"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),

            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),

            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=30000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "normal"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        loop=2,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord"},
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=30000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "normal",
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="SphereCrop", point_max=30000, mode="random"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord",),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "normal", "index"),
                    feat_keys=("coord", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1.0, 1.0])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.0, 1.0]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
