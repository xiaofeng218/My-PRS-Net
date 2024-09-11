# PRS-Net 实验复现
### 环境安装
```shell
conda create --name prsnet python==3.8.18
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
### 下载数据：
1. datasets用于保存处理好的数据  
```shell
    mkdir datasets
```
然后将下载下来的shapenet数据放到preprocess/shapenet文件夹中即可  

### 数据预处理
下载gptoolbox，并进行编译  
```shell
git clone https://github.com/alecjacobson/gptoolbox.git
cd gptoolbox/mex
mkdir build
cd build
proxychains4 cmake ..
cd ../../
```

设置matlab的路径  
```shell
export PATH=/usr/local/MATLAB/R2018a/bin:$PATH
```
因为savepath默认的文件没有权限进行修改，因此只能自己创建一个my_pathdef.m，然后将路径保存进去。在matlab中执行如下路径配置命令，让matlab能够使用gptoolbox中的文件。  
```shell
addpath(strjoin(strcat(['/mnt/2/hanxiaofeng/PRS-Net/gptoolbox/', {'external','imageprocessing', 'images', 'matrix', 'mesh', 'mex', 'quat','utility','wrappers'}]), ':'));
addpath('preprocess/point_mesh_squared_distance.mexa64')
savepath('~/my_pathdef.m')
nano /mnt/2/hanxiaofeng/PRS-Net/startup.m
run('~/my_pathdef.m')
```
启动matlab
```shell
matlab -nodesktop -nosplash
# 运行precomputeShapeData文件：在matlab命令行中输入：
preprocess/precomputeShapeData.m
```

### 训练命令
```shell
python train.py --datapath datasets/shapenet/train
```

### 测试命令
```shell
python test.py --testpath datasets/shapenet/test --modelpath logs/model/model_weights.pth
```

### 查看tensorboard上保存的训练结果
```shell
!tensorboard --logdir ./logs --port=6006 
```





