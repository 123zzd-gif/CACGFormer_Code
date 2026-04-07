# -*- coding: utf-8 -*-

import numpy as np
import glob
import os

# 设置数据路径
DATA_PATH = '/data/zuowenhao/pointnet2_my/data/s3dis/Stanford3dDataset_v1.2_Aligned_Version'
OUTPUT_PATH = '/data/zuowenhao/pointnet2_my/data/stanford_indoor3d/'  # 输出目录

# 函数：处理单个文件夹中的所有 .txt 文件，生成 .npy 文件
def collect_point_label(anno_file, out_filename, file_format='npy'):
    # 读取单个 .txt 文件
    points = np.loadtxt(anno_file)  # 读取点云数据文件，包含 XYZ 和标签
    
    # 确保文件不为空
    if points.size == 0:
        print(f"Warning: File {anno_file} is empty or has invalid data.")
        return  # 如果文件为空或无效，直接返回
    
    # 提取前三列 XYZ 坐标，并保留六位小数
    xyz = np.round(points[:, :3], 6)
    
    # 提取最后一列标签
    labels = points[:, -1].reshape(-1, 1)

    # 将标签 0 转换为 'yayin'，1 转换为 'yaguan'
    labels[labels == 0] = 0  # 'yayin' 类别为 0
    labels[labels == 1] = 1  # 'yaguan' 类别为 1

    # 设置颜色为白色 (RGB: 255, 255, 255)
    rgb = np.ones((points.shape[0], 3)) * 255  # 所有点的颜色设置为白色

    # 合并 XYZ 坐标、RGB 颜色和标签
    data = np.concatenate([xyz, rgb, labels], axis=1)  # Nx7，包含 XYZ、RGB 和标签
    
    # 根据指定的文件格式保存数据
    if file_format == 'txt':
        np.savetxt(out_filename, data, fmt='%f %f %f %d %d %d %d')  # 保存为 txt 格式
    elif file_format == 'npy':
        np.save(out_filename, data)  # 保存为 numpy .npy 文件
        print(f"File saved successfully: {out_filename}")
    else:
        print(f"ERROR: Unknown file format {file_format}. Please use 'txt' or 'npy'.")

# 函数：遍历文件夹内的每个文件，生成对应的 .npy 文件
def process_folder(folder_path, output_folder):
    for anno_file in glob.glob(os.path.join(folder_path, '*.txt')):
        elements = anno_file.split('/')
        # 根据文件路径和文件名生成输出的 .npy 文件名
        out_filename = os.path.join(output_folder, f"{elements[-2]}_{elements[-1].split('.')[0]}.npy")
        
        # 调用函数进行转换和保存
        collect_point_label(anno_file, out_filename, file_format='npy')

# 主程序：遍历 yachi 和 Test 文件夹
yachi_folder = os.path.join(DATA_PATH, 'yachi')
test_folder = os.path.join(DATA_PATH, 'Test')

# 输出路径
output_folder = OUTPUT_PATH

# 对每个文件夹进行处理
process_folder(yachi_folder, output_folder)
process_folder(test_folder, output_folder)
