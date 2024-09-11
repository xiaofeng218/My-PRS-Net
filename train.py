from viewtools import Timer, Accumulator
from torch.utils import tensorboard
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import numpy as np
from PRSNet import symLoss, regLoss, PRS_NET
from viewtools import view_result
from dataset import loadmat_dir, SeqDataLoader
import argparse

"""训练模型一个迭代周期（定义见第3章）。"""
def train_epoch(net, train_iter, sym_loss, reg_loss, updater, device, weight=25):
    state, timer = None, Timer()
    metric_loss_plane_sym = Accumulator(2)  # 统计训练损失之和
    metric_loss_plane_reg = Accumulator(2)  # 统计训练准确度之和
    metric_loss_quat_sym = Accumulator(2)  # 统计训练损失之和
    metric_loss_quat_reg = Accumulator(2)  # 统计训练准确度之和

    for Volume, surfaceSamples, closestPoints in train_iter:
        Volume = Volume.to(device)
        surfaceSamples = surfaceSamples.to(device)
        closestPoints = closestPoints.to(device)
        
        batch_size = Volume.shape[0]
        # 在第一次迭代或使用随机抽样时初始化state
        updater.zero_grad()
        # 对于ffn，我们只使用最后一个时间步计算损失
        # y_hat.shape = (num_steps * batch_size, vocab_size)
        planes, quats = net(Volume)
        loss_plane_sym, loss_quat_sym = sym_loss(planes, quats, closestPoints, surfaceSamples, Volume)
        loss_plane_reg, loss_quat_reg = reg_loss(planes, quats)
        loss = loss_plane_sym + loss_quat_sym + (loss_plane_reg + loss_quat_reg) * weight
        loss.backward()
        updater.step()
        
        metric_loss_plane_sym.add(loss_plane_sym * batch_size, batch_size)
        metric_loss_plane_reg.add(loss_plane_reg * batch_size, batch_size)
        metric_loss_quat_sym.add(loss_quat_sym * batch_size, batch_size)
        metric_loss_quat_reg.add(loss_quat_reg * batch_size, batch_size)

    return metric_loss_plane_sym[0] / metric_loss_plane_sym[1],  \
            metric_loss_plane_reg[0] / metric_loss_plane_reg[1], \
            metric_loss_quat_sym[0] / metric_loss_quat_sym[1],   \
            metric_loss_quat_reg[0] / metric_loss_quat_reg[1],   \
            metric_loss_plane_sym[1] / timer.stop()

# 将matplotlib中画出来的图像转成tensor，以便在tensorboard中进行显示
def get_tensor_from_video(video_path):
    """
    :param video_path: 视频文件地址
    :return: pytorch tensor
    """
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        else:
            # 注意，opencv默认读取的为BGR通道组成模式，需要转换为RGB通道模式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
    cap.release()
    result_frames = torch.as_tensor(np.stack(frames_list), dtype=torch.uint8)
    # 注意：此时result_frames组成的维度为[视频帧数量，宽，高，通道数]
    result_frames = result_frames.permute(0,3,2,1).unsqueeze(0)
    return result_frames

'''训练函数'''
def train(net, train_iter, lr, num_epochs, device, weight = 25, test_file_path='test/test_models_2aec'):
    """训练模型"""
    def init_weights(layer):
        if isinstance(layer, nn.Conv3d):
            # 使用 Kaiming 正态分布初始化
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            # 使用 Xavier 均匀分布初始化
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    net.to(device)  
    net.apply(init_weights)

    sym_loss = symLoss(device)
    reg_loss = regLoss(device)

    writer = tensorboard.SummaryWriter(log_dir="./logs", filename_suffix='prsnet')
    test_input = torch.rand((32, 32, 32, 32)).to(device)
    writer.add_graph(net, test_input) 

    updater = torch.optim.Adam(net.parameters(), lr)

    test_data = loadmat_dir(test_file_path)
    test_Volume = test_data['Volumes'][0]
    test_Volume = torch.tensor(test_Volume).to(torch.float32).unsqueeze(0).to(device)

    # 训练和预测
    for epoch in tqdm(range(num_epochs)):
        loss_plane_sym, loss_plane_reg, loss_quat_sym, loss_quat_reg, speed = train_epoch(
                                                        net, train_iter, sym_loss, reg_loss, updater, device, weight)
        # if (epoch + 1) % 10 == 0:
        writer.add_scalars("sym_loss", {'plane_loss': loss_plane_sym}, epoch)
        writer.add_scalars("sym_loss", {'quat_loss': loss_quat_sym}, epoch)
        writer.add_scalars("reg_loss", {'plane_loss': loss_plane_reg}, epoch)
        writer.add_scalars("reg_loss", {'quat_loss': loss_quat_reg}, epoch)
        
        if epoch % 40 == 0:
            with torch.no_grad():
                planes, quats = net(test_Volume)
            save_path = "test/images/train_"+str(epoch)+'.mp4'
            view_result(save_path, None, test_Volume[0], None, None, planes, quats)
            vedio = get_tensor_from_video(save_path)
            writer.add_video("Animation", vedio, epoch) 
            
    print(f'loss_plane_sym: {loss_plane_sym:.1f},\n\
          loss_plane_reg: {loss_plane_reg:.1f}, \n\
          loss_quat_sym: {loss_quat_sym:.1f}, \n\
          loss_quat_reg: {loss_quat_reg:.1f}, \n\
          {speed:.1f} 模型/秒 {str(device)}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PRSNet Arguments")
    parser.add_argument('--datapath', type=str, help='输入训练数据地址')

    args = parser.parse_args()

    lr, num_epochs, weight = 0.001, 200, 25
    batch_size = 32

    # 测试网络结构
    file_path = 'datasets/shapenet/train'
    net = PRS_NET()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print("Training on:", device)
    print("Loading data")
    data_loader = SeqDataLoader(args.datapath, batch_size=batch_size, shuffle=True)
    print("Start Training")
    train(net, data_loader, lr, num_epochs, device, weight)
    torch.save(net.state_dict(), './logs/model/model_weights.pth')
