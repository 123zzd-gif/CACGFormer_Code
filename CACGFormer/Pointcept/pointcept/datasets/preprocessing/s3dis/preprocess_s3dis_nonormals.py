# -*- coding: utf-8 -*-

import os
import argparse
import glob
import numpy as np

try:
    import open3d
except ImportError:
    import warnings

    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings

    warnings.warn("Please install trimesh for parsing normal")

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def parse_room(room, dataset_root, output_root, align_angle=True, parse_normal=False):
    print(f"Parsing: {room}")

    
    class2label = { #"erchuiqu": 0,
                    # "erjiaqu": 1,
                    # "erpingqu": 2,
                    # "duierpingqu": 3,
                    # "erzhouqu": 4,
                    # "sanjiaowoqu": 5,
                    # "duierlunqu": 6,
                    # "erlunqu": 7,
                    # "shen": 8,
                    # "yidan": 9,
                    # "gan": 10,
                    # "pi": 11,
                    # "fei": 12,
                    # "xin": 13,
                    # "sanjiao": 14,
                    # "neifenmi":15
                    "beijing": 0,
                    "erjiaqu": 1,
                    "shen": 2,
                    "yidan": 3,
                    "gan": 4,
                    "pi": 5,
                    "fei": 6,
                    "xin": 7,
                    "sanjiao": 8,
                    "neifenmi":9
                   }  # 牙龈：0，牙冠：1   {"yayin": 0, "yaguan": 1}

    source_dir = os.path.join(dataset_root, room)
    save_path = os.path.join(output_root, room)
    os.makedirs(save_path, exist_ok=True)

    # 获取该房间下的所有 txt 文件路径
    object_path_list = sorted(glob.glob(os.path.join(source_dir, "*.txt")))

    for object_path in object_path_list:
        # 读取txt文件
        obj = np.loadtxt(object_path)

        # 分割坐标和类别（去除后三列法向量）
        coords = obj[:, :3]  # 坐标（前三列）
        colors = np.ones_like(coords) * 255  # 默认为白色
        labels = obj[:, -1].astype(int)  # 假设最后一列是类别标签

        # 将labels作为实例ID，生成instance.npy
        instances = labels  # 如果instance与segment相同，可以直接使用labels作为实例ID

        # 保存每个txt文件的对应数据
        object_name = os.path.basename(object_path).split(".")[0]  # 用文件名作为识别
        file_save_path = os.path.join(save_path, object_name)

        os.makedirs(file_save_path, exist_ok=True)  # 为每个牙齿创建一个子文件夹

        # 保存为Numpy文件
        np.save(os.path.join(file_save_path, "coord.npy"), coords.astype(np.float32))
        np.save(os.path.join(file_save_path, "color.npy"), colors.astype(np.uint8))  # 保存白色颜色
        np.save(os.path.join(file_save_path, "segment.npy"), labels.astype(np.int16))  # 保存标签
        np.save(os.path.join(file_save_path, "instance.npy"), instances.astype(np.int16))  # 保存实例 ID


def main_process():
    dataset_root = "/dataset/zuowenhao/pointnet2_my/data/ear_seg_3D_2xuequ_small_xuequ/"  # 数据集路径"/data/zuowenhao/PointTransformerV3_1/datateeth/teeth_data_for_new/000001/""/data/zuowenhao/pointnet2_my/data/ear_seg_3D/"
    output_root = "/dataset/zuowenhao/PointTransformerV3_1/dataset/s3dis/"  # 输出路径

    room_list = ["train", "val"]  # 房间目录（训练、验证、测试集）["train", "test", "val"] ["ear", "Test"]

    for room in room_list:
        parse_room(room, dataset_root, output_root)


if __name__ == "__main__":
    main_process()