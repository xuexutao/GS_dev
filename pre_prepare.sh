echo "==========================start prepare environment=========================="
echo "config cache and cuda ... "
sudo rm -rf ~/.cache
sudo rm /usr/local/cuda
sudo ln -s /mnt/bn/isp-traindata-lf-2/xxt/.cache/ ~/.cache
sudo ln -s /mnt/bn/isp-traindata-lf-2/xxt/mlx_devbox/users/xuexutao/playground/download/cuda-12.1/ /usr/local/cuda
export HTTP_PROXY=http://bj-rd-proxy.byted.org:8118
export http_proxy=http://bj-rd-proxy.byted.org:8118
export https_proxy=http://bj-rd-proxy.byted.org:8118

echo "==========================install miniconda ... "
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "==========================create conda environment python == 3.10 ... "
conda create python=3.10 -n gaussian -y
conda activate gaussian
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install "pip<24"
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install submodules/fused-ssim/
pip install plyfile opencv-python
sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0
pip install tqdm
pip install "numpy<2"

echo "==========================success install=========================="





# =======================================================
# git remote add origin https://github.com/xuexutao/GS_dev.git
# git branch -M main
# git push -u origin main
# ===================git config==========================
# git config --global user.name "xuexutao"
# git config --global user.email "645193045@qq.com"
# 新建一个ssh key
# ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_xuexutao -C "645193045@qq.com"
# cat ~/.ssh/id_ed25519_xuexutao.pub
# 复制到: GitHub → Settings → SSH and GPG keys → New SSH key
# vim ~/.ssh/config
    # 添加
    # Host github-xuexutao
    #     HostName github.com
    #     User git
    #     IdentityFile ~/.ssh/id_ed25519_xuexutao
# 测试连接
# git remote set-url origin git@github-xuexutao:xuexutao/GS_dev.git
# ssh -T git@github-xuexutao
# 成功后，会显示：
# Hi xuexutao! You've successfully authenticated, but GitHub does not provide shell access.
# git push -u origin main
# =======================================================