# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/zwwjava/HS-Restormer
cd HS-Restormer
```

2. Make conda environment
```
conda create -n pytorch181 python=3.7
conda activate pytorch181
```

3. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
# 看情况单独安装tb-nightly
pip install tb-nightly==2.12.0a20230113  -i https://mirrors.aliyun.com/pypi/simple
pip install tb_nightly==2.14.0a20230808 -i https://mirrors.aliyun.com/pypi/simple
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

