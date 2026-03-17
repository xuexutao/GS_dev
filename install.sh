conda create -n gs python=3.10
conda activate gs
pip install "pip<24"
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install submodules/fused-ssim/
pip install plyfile opencv-python

sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0

pip install tqdm
pip install 'numpy<2'

