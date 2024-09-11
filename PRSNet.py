import torch
from torch import nn
import math

class MyLinear(nn.Module):
    def __init__(self, n, in_features, out_features):
        super(MyLinear, self).__init__()
        self.n = n
        self.weight = nn.Parameter(torch.randn(n, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n, out_features))

    def forward(self, x):
        # x 的形状为 (b, n, in) 或者 (b, n * in)
        # 批量矩阵乘法，n 个不同的全连接层应用到对应的 n 维度的子向量上
        # torch.einsum 执行 (b, n, in) @ (n, in, out) -> (b, n, out)
        x = x.view(x.size(0), self.n, -1)
        output = torch.einsum('bni,nio->bno', x, self.weight)
        # 加上偏置，使用广播机制，偏置的形状为 (n, out)
        output = output + self.bias
        return output
    
class PRS_NET(nn.Module):
    # input_size: 输入体素的分辨率大小
    # output_size: 输出的平面和四元数的个数
    def __init__(self, intput_size=32, output_size=3):
        super(PRS_NET, self).__init__()
        # The CNN has five 3D convolution layers of 
        # kernel size 3, padding 1, and stride 1. 
        # After each 3D convolution, a max pooling of kernel size 2 
        # and leaky ReLU activation are applied.
        self.output_size = output_size
        conv_layer_depth = 6
        conv = nn.Sequential()
        conv.add_module(f'conv0', nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1))
        conv.add_module(f'pool0', nn.MaxPool3d(kernel_size=2))
        conv.add_module(f'leaky_relu0', nn.LeakyReLU())
        for i in range(2, conv_layer_depth):
            conv.add_module(f'conv{i-1}', nn.Conv3d(in_channels=2**i, out_channels=2**(i+1), kernel_size=3, padding=1))
            conv.add_module(f'pool{i-1}', nn.MaxPool3d(kernel_size=2))
            conv.add_module(f'leaky_relu{i-1}', nn.LeakyReLU())
        self.conv = conv

        # fnn的各层大小默认为32、16、4
        fnn_num = output_size * 2
        fnn = nn.Sequential()
        fnn.add_module(f'fnn{i}_layer0', nn.Linear(64, 32 * fnn_num))
        fnn.add_module(f'fnn{i}_leaky_relu0', nn.LeakyReLU())
        fnn.add_module(f'fnn{i}_layer1', MyLinear(fnn_num, 32, 16))
        fnn.add_module(f'fnn{i}_leaky_relu1', nn.LeakyReLU())
        fnn.add_module(f'fnn{i}_layer2', MyLinear(fnn_num, 16, 4))
        self.fnn = fnn
        self.fnn_num = fnn_num

    def forward(self, volume):
        # volume: (batch_size, 32, 32, 32)
        # print("volume:", volume.size)
        # output: plane, quat: (batch_size, 4)
        volume = volume.unsqueeze(1)
        # volume: (batch_size, 1, 32, 32, 32)
        conv_output = self.conv(volume)
        # print("conv_output:", conv_output.size)
        # conv_output.size = (batch_size, 32, 1, 1, 1)
        # flatten.size = (batch_size, 32)
        flatten = conv_output.view(conv_output.size(0), -1)
        # print("flatten:", flatten.size)
        # fnn_output.size = (output_size * 2, batch_size, 4)
        fnn_output = self.fnn(flatten).permute(1,0,2)
        # print("fnn_output:", fnn_output.size)

        planes = fnn_output[:3]
        quats = fnn_output[3:]

        # planes.shape: (output_size, batch_size, 4)
        # quats.shape: (output_size, batch_size, 4)
        return planes, quats

# 体素坐标系与点云坐标系之间的转换：(0 - 32)以及(-0.5 - 0.5)
def point2voxel(px, gridSize=32, gridBound=0.5):
    gridMin = -gridBound + gridBound / gridSize
    vx = (px - gridMin) * gridSize / (2 * gridBound)
    return vx

def voxel2point(x, gridSize=32, gridBound=0.5):
    gridMin = -gridBound + gridBound / gridSize
    return x * (2 * gridBound) / 32 + gridMin

"""四元数乘法，支持批量操作"""
def quaternion_multiply(q1, q2):
    # q1.shape = (batch, 1, 4)
    # q2.shape = (batch, 1000, 4)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

"""四元数共轭"""
def quaternion_conjugate(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([w, -1 * x, -1 * y, -1 * z], dim=-1)

"""使用四元数q旋转点p，支持批量操作"""
def rotate_point(p, q):
    # p.shape = (batch, 1000, 3)
    # q.shape = (batch, 1, 4)
    p_quat = torch.cat([torch.zeros(p.shape[0], p.shape[1], 1, device=p.device), p], dim=-1)
    q_conj = quaternion_conjugate(q)
    p_rotated = quaternion_multiply(quaternion_multiply(q, p_quat), q_conj)
    return p_rotated[..., 1:]  # 返回旋转后的点坐标

"""计算点关于一个平面的对称点"""
def plane_point(p, plane):
    # p.shape = (batch, 1000, 3)
    # plane.shape = (batch, 1, 4)
    length = 2 * (torch.sum(p * plane[...,:3], dim=2, keepdim=True) + plane[...,3:4]) # (batch_size, 1000, 1)
    norm2 = torch.sum(plane[..., :3] ** 2, dim=2, keepdim=True) # (batch_size, 1, 1)
    p1 = p - length * plane[...,:3].repeat(1, length.shape[1], 1) / (norm2 + 1e-8) # (batch_size, 1000, 3)
    return p1

"""计算点到最近点的距离，输入为点云坐标单位，输出也为点云坐标单位"""
def calculate_distance(points, closestPoints, volume, device, gridSize=32):
    # points.shape = (batch_size, 1000, 3)
    # closestPoints.shape = (batch_size, 32, 32, 32, 3)
    # volume.shape = (batch_size, 32, 32, 32)
    # print(points.shape, closestPoints.shape, volume.shape)
    # torch.Size([1, 1000, 3]) torch.Size([1, 32, 32, 32, 3]) torch.Size([1, 32, 32, 32])

    # 寻找这个点所在的网格编号
    inds = point2voxel(points)
    # inds.shape = (batch_size, 1000, 3)
    inds = torch.round(torch.clamp(inds, min=0, max=gridSize-1))
    # inds.shape = (batch_size, 1000)
    inds = torch.matmul(inds, torch.FloatTensor([gridSize**2, gridSize, 1]).to(device)).long()
    # v.shape = (batch_size, 32*32*32)
    v = volume.view(-1, gridSize**3)
    # 这里需要对距离计算一个mask，因为有些点对应的网格是有像素的，因此这时候再算距离就不准确了，因此需要将这些点的距离置为0
    # mask.shape = (batch_size, 1000, 1)
    mask = (1 - torch.gather(v, 1, inds)).unsqueeze(2)
    inds = inds.unsqueeze(2).repeat(1, 1, 3)
    # inds.shape = (batch_size, 1000, 3)
    cps = closestPoints.reshape(closestPoints.shape[0], -1, 3) # (batch_size, 32*32*32, 3)
    cps = torch.gather(cps, 1, inds).to(device) 
    # cps.shape = (batch_size, 1000, 3)
    # ------------
    return (points - cps) * mask, cps * mask

"""计算对称性损失"""
def sym_loss(planes, quats, closestPoints, surfaceSamples, volume, device):
    # planes.shape = (output_size, batch_size, 4)
    # quats.shape = (output_size, batch_size, 4)
    # closestPoints.shape = (batch_size, 32, 32, 32, 3)
    # surfaceSamples.shape = (batch_size, 1000, 3)
    loss_planes_sym = 0
    loss_quats_sym = 0
    for i in range(planes.size(0)):
        plane = planes[i].unsqueeze(1).float() # (batch_size, 1, 4)
        quat = quats[i].unsqueeze(1).float() # (batch_size, 1, 4)

        sym_Points_plane = plane_point(surfaceSamples, plane) # (batch_size, 1000, 3)
        distance, _ = calculate_distance(sym_Points_plane, closestPoints, volume, device) # (batch_size, 1000, 3)
        loss_planes_sym += torch.mean(torch.sum(torch.norm(distance, dim=2), dim=1))

        sym_Points_quat = rotate_point(surfaceSamples, quat)
        distance, _ = calculate_distance(sym_Points_quat, closestPoints, volume, device)
        loss_quats_sym += torch.mean(torch.sum(torch.norm(distance, dim=2), dim=1))

    return loss_planes_sym / planes.size(0), loss_quats_sym / quats.size(0)

"""计算正则化损失""" 
def reg_loss(planes, quats, device):
    # planes.shape = (output_size, batch_size, 4)
    # quats.shape = (output_size, batch_size, 4)
    eye = torch.eye(3).unsqueeze(0).to(device)
    M1 = planes[..., :3].permute(1, 0, 2) # (batch_size, 3, 3)
    # 对M的列向量做归一化
    M1 = M1.div(torch.norm(M1, dim=2, keepdim=True) + 1e-8)
    M1_T = M1.permute(0, 2, 1) # (batch_size, 3, output_size)
    loss_planes_reg = (torch.matmul(M1, M1_T) - eye).pow(2).sum(2).sum(1).mean()
    
    M2 = quats[..., 1:4].permute(1, 0, 2) # (batch_size, output_size, 3)
    M2 = M2.div(torch.norm(M2, dim=2, keepdim=True) + 1e-8)
    M2_T = M2.permute(0, 2, 1) # (batch_size, 3, output_size)
    loss_quats_reg = (torch.matmul(M2, M2_T) - eye).pow(2).sum(2).sum(1).mean()

    return loss_planes_reg, loss_quats_reg

class symLoss(nn.Module):
    def __init__(self, device):
        super(symLoss, self).__init__()
        self.device = device
    def forward(self, planes, quats, closestPoints, surfaceSamples, volume):
        # 定义损失计算逻辑
        # 例如，使用均方误差作为损失
        return sym_loss(planes, quats, closestPoints, surfaceSamples, volume, self.device)

class regLoss(nn.Module):
    def __init__(self, device):
        super(regLoss, self).__init__()
        self.device = device    
    def forward(self, planes, quats):
        # 定义损失计算逻辑
        # 例如，使用均方误差作为损失
        return reg_loss(planes, quats, self.device)