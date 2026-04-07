import numpy as np
import os

# Set the file paths
base_coord_path = "/home/m/process/PointTransformerV3_1/dataset/s3dis_2fenlei_8xiaoxueqv_normals/test/"
base_pred_path = "/home/m/process/PointTransformerV3_1/Pointcept/tools/exp/default_ear_xi__keshihua_beijing+erjia+8xiaoxuequ_normals_150guanfang+96zijicaiji_3jihetezheng+loss_zhuyili/result/epoch_40"#/data/zuowenhao/PointTransformerV3_1/Pointcept/tools/exp/default/result/epoch_20/"
output_path = "/home/m/process/PointTransformerV3_1/keshihua_xincaijian_ear_二分类+8小穴区——normals创新点——3jihetezheng_zhuyili/"

# Loop through the 26 folders (yachi_1 to yachi_26)
for i in range(120,597):
    # Set the path for coord.npy file
    coord_path = os.path.join(base_coord_path, f"{i}_s", "coord.npy")
    
    # Load the coord.npy file
    if os.path.exists(coord_path):
        coords = np.load(coord_path)  # Shape: (N, 3)
    else:
        print(f"Warning: {coord_path} does not exist, skipping this folder.")
        continue
    
    # Set the corresponding prediction file path
    pred_file = os.path.join(base_pred_path, f"test-{i}_s_pred.npy")
    
    # Load the prediction labels file
    if os.path.exists(pred_file):
        labels = np.load(pred_file)  # Shape: (N,)
    else:
        print(f"Warning: {pred_file} does not exist, skipping this file.")
        continue
    
    # Check if the number of points and labels match
    if len(coords) == len(labels):
        # Combine coordinates and labels data
        combined_data = np.hstack((coords, labels.reshape(-1, 1)))  # Shape: (N, 4)
        
        # Save the combined data to a .txt file
        output_file = os.path.join(output_path, f"{i}_s.txt")
        np.savetxt(output_file, combined_data, fmt="%f", delimiter=" ")
    else:
        print(f"[Mismatch] File {i}_s: coords.shape = {coords.shape}, labels.shape = {labels.shape}")
        print(f"Warning: The number of coordinates and labels do not match for file {i}, skipping this file.")

