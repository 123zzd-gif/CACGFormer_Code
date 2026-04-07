import os
import numpy as np
import pickle
import torch
import importlib
from torch.utils.data import Dataset
from tqdm import tqdm

batch_size = 16         
num_worker = 8
batch_size_test = 8
batch_size_val = 8
empty_cache = False
enable_amp = False
eval_epoch = 1  
seed = 42
resume = False
save_path = "/data/zuowenhao/PointTransformerV3_1/log/classification/pointv3_cls"
sync_bn = False
find_unused_parameters = False 

model = dict(
    type="DefaultClassifier",
    num_classes=208,
    backbone_embed_dim=512,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),####(2, 2, 2, 6, 2)
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),####(2, 4, 8, 16, 32)
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        #dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 1000
mix_prob = 0.1
evaluate = False
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.0001)
scheduler = dict(
    type="OneCycleLR",   
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0               
) 

param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "ModelNetDataset"
data_root = "/data/zuowenhao/PointTransformerV3_1/datateeth/modelnet40_normal_resampled/"
cache_data = False
class_names = [
    "36716602l",
    "36716602u",
    "36775520l",
    "36775520u",
    "36991422l",
    "36991422u",
    "37001636l",
    "37001636u",
    "37002415l",
    "37002415u",
    "37005019l",
    "37005019u",
    "37099548l",
    "37099548u",
    "37099622l",
    "37099622u",
    "37099680l",
    "37099680u",
    "37099725l",
    "37099725u",
    "37099809l",
    "37099809u",
    "37186541l",
    "37186541u",
    "37186552l",
    "37186552u",
    "37341518l",
    "37341518u",
    "37341528l",
    "37341528u",
    "37342816l",
    "37342816u",
    "37342894l",
    "37342894u",
    "37343017l",
    "37343017u",
    "37343124l",
    "37343124u",
    "37343228l",
    "37343228u",
    "37343425l",
    "37343425u",
    "37397563l",
    "37397563u",
    "37398184l",
    "37398184u",
    "37424773l",
    "37424773u",
    "37425237l",
    "37425237u",
    "37437857l",
    "37437857u",
    "37438973l",
    "37438973u",
    "37444192l",
    "37444192u",
    "37445079l",
    "37445079u",
    "37446266l",
    "37446266u",
    "37447008l",
    "37447008u",
    "37499297l",
    "37499297u",
    "37499330l",
    "37499330u",
    "37499928l",
    "37499928u",
    "37666825l",
    "37666825u",
    "37667402l",
    "37667402u",
    "37668856l",
    "37668856u",
    "62864963l",
    "62864963u",
    "62865074l",
    "62865074u",
    "62865182l",
    "62865182u",
    "62865278l",
    "62865278u",
    "62865359l",
    "62865359u",
    "62865446l",
    "62865446u",
    "63379017l",
    "63379017u",
    "65396887l",
    "65396887u",
    "66633465l",
    "66633465u",
    "66633703l",
    "66633703u",
    "66633887l",
    "66633887u",
    "66634455l",
    "66634455u",
    "66635201l",
    "66635201u",
    "66636212l",
    "66636212u",
    "66637536l",
    "66637536u",
    "66638441l",
    "66638441u",
    "66639854l",
    "66639854u",
    "66839762l",
    "66839762u",
    "66840064l",
    "66840064u",
    "66840432l",
    "66840432u",
    "66840720l",
    "66840720u",
    "66841044l",
    "66841044u",
    "66841294l",
    "66841294u",
    "66841684l",
    "66841684u",
    "66842042l",
    "66842042u",
    "66842400l",
    "66842400u",
    "66842585l",
    "66842585u",
    "66842888l",
    "66842888u",
    "66843218l",
    "66843218u",
    "66843461l",
    "66843461u",
    "66843690l",
    "66843690u",
    "66843963l",
    "66843963u",
    "66844153l",
    "66844153u",
    "66844367l",
    "66844367u",
    "66844567l",
    "66844567u",
    "66844878u",
    "66845102l",
    "66845102u",
    "66845364l",
    "66845364u",
    "66845371l",
    "66845371u",
    "66845632l",
    "66845632u",
    "66860646l",
    "66860646u",
    "66864923l",
    "66864923u",
    "66865139l",
    "66865139u",
    "66950279l",
    "66950279u",
    "66950349l",
    "66950349u",
    "66950512l",
    "66950512u",
    "66950694l",
    "66950694u",
    "66950843l",
    "66950843u",
    "66950977l",
    "66950977u",
    "66951123l",
    "66951123u",
    "66951443l",
    "66951443u",
    "66953634l",
    "66953634u",
    "66954328l",
    "66954328u",
    "66954811l",
    "66954811u",
    "67116356l",
    "67116356u",
    "67117103l",
    "67117103u",
    "67120397l",
    "67120397u",
    "67121538l",
    "67121538u",
    "67122463l",
    "67122463u",
    "67123330l",
    "67123330u",
    "67124716l",
    "67124716u",
    "67287038l",
    "67287038u",
    "67287794l",
    "67287794u",
    "67288373l",
    "67288373u",
    "67289173l",
    "67289173u",
    "67291632l",
    "67291632u",
    "67292844l",
    "67292844u",
    "67293120l",
    "67293120u",

]

data = dict(
    num_classes=208,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=10000, mode="random"),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "normal"),
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "normal"],
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],  # 1
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 2
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 3
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 4
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 6
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 7
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 8
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 9
            ],
        ),
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="ClsVotingTester", num_repeat=100)


test["batch_size"] = batch_size_test 
train = dict(
    type="DefaultTrainer",
    cfg="cls-ptv3-v1m1-0-base.py",  
)
