import torch
from PRSNet import sym_loss, PRS_NET
from viewtools import view_result
from tqdm import tqdm
from dataset import loadmat_dir
import argparse

# 清除掉太过于重合或者置信度太低的平面和旋转向量
def validation(planes, quats, closestPoints, surfaceSamples, volume, device, eps=4e-4):
    valid_plane = torch.ones(3, dtype=torch.bool).to(device)
    valid_quat = torch.ones(3, dtype=torch.bool).to(device)
    plane_losses, quat_losses = [], []

    # 排除损失值太大的对称平面或旋转轴
    for i in range(planes.shape[0]):
        plane_loss, quat_loss = sym_loss(planes[i].unsqueeze(0), 
                                         quats[i].unsqueeze(0), closestPoints, surfaceSamples, volume, device)
        if plane_loss > eps:
            valid_plane[i] = 0
        if quat_loss > eps:
            valid_quat[i] = 0
        
        plane_losses.append(plane_loss)
        quat_losses.append(plane_loss)
    
    print("Plane_losses:", plane_losses)
    print("quat_losses:", quat_loss)
    
    # 排除彼此之间靠的太近的对称平面或旋转轴
    def test_angle(vec1, vec2):
        angle = torch.dot(vec1, vec2) / torch.sqrt(torch.norm(vec1) * torch.norm(vec2))
        if torch.abs(angle) > (torch.sqrt(3) / 2):
            return 1
        return 0

    def remove_overlap_vec(vecs, valid_vec, vec_losses):
        for i in range(vecs.shape[0]):
            for j in range(0, i):
                if valid_vec[i] == 0 or valid_vec[j] == 0:
                    continue
                if test_angle(vecs[i], vecs[j]) == 1:
                    if vec_losses[i] > vec_losses[j]:
                        valid_vec[i] = 0
                    else:
                        valid_vec[j] = 0
    
    remove_overlap_vec(planes[..., :3], valid_plane, plane_losses)
    remove_overlap_vec(quats[..., :3], valid_quat, quat_losses)

    valid_planes = planes[valid_plane]
    valid_quats = quats[valid_quat]

    return valid_planes, valid_quats

# 输入Volume，输出预测出来的对称平面视频到save_path
def predict(net, obj_path, save_path, Volume, closestPoints, surfaceSamples, device):
    '''
        net: 网络
        obj_path: 物体的原始网格表示
        save_path: 保存视频的地址
        Volume: 物体的体素表示，(1, 32, 32, 32)
        closestPoints: 最近点，(1, 32, 32, 32, 3)
        surfaceSamples: 物体表面的采样点，(1, 1000, 3)
    '''
    Volume = Volume.to(torch.float32).to(device)
    closestPoints = closestPoints.to(device)
    surfaceSamples = surfaceSamples.to(device)
    # print("surfaceSamples.shape", surfaceSamples.shape)
    with torch.no_grad():
        planes, quats = net(Volume)
    # 由于实际计算出来的损失值太大，按照文章中的eps值，会将所有的对称平面和对称轴给删除掉
    # valid_planes, valid_quats = validation(planes, quats, closestPoints, surfaceSamples, Volume, device)
    # sym_Points_plane = plane_point(surfaceSamples, planes[1:2])
    # distance, cps = calculate_distance(sym_Points_plane, closestPoints, Volume, device)
    # view_result(save_path, None, Volume[0], None, cps[0], planes[1:2], None)
    view_result(save_path, None, Volume[0], None, None, planes, None)
    return planes, quats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PRSNet Arguments")
    parser.add_argument('--testpath', type=str, help='输入测试数据地址')
    parser.add_argument('--modelpath', type=str, help='输入测试模型地址')
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = PRS_NET()
    net = net.to(device)
    net.load_state_dict(torch.load(args.modelpath))
    # test_path = 'datasets/shapenet/test'
    data = loadmat_dir(args.testpath)
    for i in tqdm(range(0, 20)):
        save_path = "test/images/predict" + str(i) + ".mp4"
        predict(net, None, save_path, data['Volumes'][i:i+1], data['closestPoints'][i:i+1], data['surfaceSamples'][i:i+1], device)
    print('done')
