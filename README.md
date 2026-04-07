Category-aware and Centroid-Guided Point Transformer for Precise 3D Ear Region Segmentation
<img width="553" height="284" alt="image" src="https://github.com/user-attachments/assets/92f5298f-8747-423c-8a71-f05d10f6676d" />
Installation
Requirements
Ubuntu: 20.04 and above
CUDA: 11.6 and above
PyTorch: 1.12.0 and above
Environment
Base environment
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric
cd libs/pointops
python setup.py install
cd ../..
pip install spconv-cu118 
pip install open3d
