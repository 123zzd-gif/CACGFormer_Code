_base_ = ["../_base_/default_runtime.py"]
save_path = "exp/default_ear_xi__weight+houchuli"
# misc custom setting
batch_size = 2  # bs: total bs in all gpus
num_worker = 16
mix_prob = 0.8
empty_cache = True
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=16,
    backbone_out_channels=544,# chushi:64,1024

    # 新增预测后处理参数三个
    # postprocess_inference_only=True, #在测试时应用后处理
    # enable_prediction_postprocess=True,  # 启用预测后处理
    # sparse_indices=range(8, 16),  # 稀疏类别索引(8-15)
    # k_nearest=25,  # 每个质心保留最近的25个点
    ###

    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),

        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0, #0.0,
        proj_drop=0.0, #0.0,
        drop_path=0.3, #0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
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
        sparse_indices=range(8, 16),  # 明确指定稀疏类别索引(8-15)
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        #dict(type="FocalLoss", gamma=2.0, alpha=0.5, loss_weight=0.15, ignore_index=-1),
        #dict(type="DiceLoss", loss_weight=0.5, smooth=1e-5, ignore_index=-1),
        #  dict(
        #     type="DynamicClassAwareLoss", 
        #     num_classes=16,
        #     base_weight=1.0, 
        #     sparse_weight=2.5,
        #     sparse_indices=range(8, 16),  # 明确指定稀疏类别索引(8-15)
        #     ignore_index=-1,
        #     loss_weight=0.5
        # ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
            # 新增：类别平衡正则化损失
        dict(
            type="ClassBalancedRegularizationLoss", 
            weight_reg_scale=0.01,  # 正则化强度，可以调整
            loss_weight=0.1,        # 相对于其他损失的权重
            ignore_index=-1
        ),
    ],
)

#load_from = "/dataset/zuowenhao/PointTransformerV3_1/Pointcept/tools/exp/default_ear_xi_9/model/model_best.pth"
# scheduler settings
epoch = 600
eval_epoch = 300
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)#weight_decay=0.0005
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.015, 0.00005],   #[0.005, 0.0005]
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=1.0,###10.0
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "S3DISDataset"
data_root = "/dataset/zuowenhao/PointTransformerV3_1/dataset/s3dis/"

data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        "erchuiqu",
        "erjiaqu",
        "erpingqu",
        "duierpingqu",
        "erzhouqu",
        "sanjiaowoqu",
        "duierlunqu",
        "erlunqu",
        "shen",
        "yidan",
        "gan",
        "pi",
        "fei",
        "xin",
        "sanjiao",
        "neifenmi"
    ],
    train=dict(
        type=dataset_type,
        split=("train-400"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),######yuanshi
            #dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            #dict(type="RandomRotate", angle=[-2, 2], axis="x", center=[0, 0, 0], p=0.5),###zengjia
            #dict(type="RandomRotate", angle=[-2, 2], axis="y", center=[0, 0, 0], p=0.5),###zengjia
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),######yuanshi
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2), ###zengjia
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            #dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),######yuanshi
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),######yuanshi
            dict(type="RandomJitter", sigma=0.005, clip=0.02),######yuanshi
            #dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),###zengjia
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,#grid_size=0.02
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=30000, mode="random"),
            dict(type="CenterShift", apply_z=False),######yuanshi
            #dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord"),
            ),
          
        ],
        test_mode=False,
        loop=2),
    val=dict(
        type=dataset_type,
        split="val-100",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),######yuanshi
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.02,#grid_size=0.02
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),######yuanshi
            #dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "segment",
                    "origin_segment",
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val-100",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            #dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,#grid_size=0.02
                hash_type="fnv",
                mode="test",
                keys=("coord"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
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
                    dict(type="RandomScale", scale=[1, 1]),
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
