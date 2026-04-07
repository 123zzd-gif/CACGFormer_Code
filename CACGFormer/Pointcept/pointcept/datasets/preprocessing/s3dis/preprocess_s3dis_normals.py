# -*- coding: utf-8 -*-
import os
import argparse
import glob
import numpy as np

# 无需open3d/trimesh，直接读取txt内法向量，保留导入仅做兼容（可直接删除）
try:
    import open3d
except ImportError:
    pass
try:
    import trimesh
except ImportError:
    pass

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def parse_room(room, dataset_root, output_root, align_angle=False):
    """
    处理单个分区数据，直接读取txt中xyz+法向量+标签
    txt格式：x y z nx ny nz label
    """
    print(f"Parsing: {room}")

    class2label = {
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
    }

    source_dir = os.path.join(dataset_root, room)
    save_path = os.path.join(output_root, room)
    os.makedirs(save_path, exist_ok=True)

    # 获取该分区下的所有 txt 文件路径
    object_path_list = sorted(glob.glob(os.path.join(source_dir, "*.txt")))

    for object_path in object_path_list:
        # 读取txt文件：xyz(0-2) + normal(3-5) + label(最后1列)
        obj = np.loadtxt(object_path)
        # 直接切分对应列，核心修改点
        coords = obj[:, 0:3].astype(np.float32)    # xyz坐标
        normals = obj[:, 3:6].astype(np.float32)   # 法向量nxnyz
        labels = obj[:, -1].astype(int)            # 最后一列标签

        colors = np.ones_like(coords) * 255  # 保留原有白色颜色逻辑
        instances = labels                    # 保留原有实例ID=标签的逻辑

        # 保存路径（原有逻辑，每个txt对应一个子文件夹）
        object_name = os.path.basename(object_path).split(".")[0]
        file_save_path = os.path.join(save_path, object_name)
        os.makedirs(file_save_path, exist_ok=True)

        # 保存所有数据（新增normal.npy）
        np.save(os.path.join(file_save_path, "coord.npy"), coords)
        np.save(os.path.join(file_save_path, "color.npy"), colors.astype(np.uint8))
        np.save(os.path.join(file_save_path, "segment.npy"), labels.astype(np.int16))
        np.save(os.path.join(file_save_path, "instance.npy"), instances.astype(np.int16))
        np.save(os.path.join(file_save_path, "normal.npy"), normals)  # 保存法向量


def main_process():
    # 简单命令行参数（可直接改默认值，也可命令行传参）
    parser = argparse.ArgumentParser(description="处理点云TXT：xyz+法向量+标签，保存法向量")
    parser.add_argument("--dataset_root", 
                        default="/home/m/process/PointTransformerV3_1/dataset/beijing+erjiaqv_8xiaoqv_guanfang+zijicaiji",
                        help="原始TXT数据集根路径")
    parser.add_argument("--output_root", 
                        default="/home/m/process/PointTransformerV3_1/dataset/s3dis_2fenlei_8xiaoxueqv_normals/",
                        help="处理后数据保存根路径")
    parser.add_argument("--num_workers", type=int, default=1, help="多进程数，加速处理")
    args = parser.parse_args()

    room_list = ["train", "val", "test"]  # 你的数据集分区

    # 多进程/串行处理（保留原有逻辑，可加速）
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            pool.map(parse_room,
                     room_list,
                     repeat(args.dataset_root),
                     repeat(args.output_root))
    else:
        for room in room_list:
            parse_room(room, args.dataset_root, args.output_root)


if __name__ == "__main__":
    main_process()