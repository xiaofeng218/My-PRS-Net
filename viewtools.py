import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import trimesh
import time
import os
from PRSNet import point2voxel, voxel2point

'''用于可视化结果'''
def view_result(save_path, obj_path=None, volume=None, points=None, sym_points=None, planes=None, quats=None, gridSize=32):
    '''
        save_path: 保存可视化结果的地址  
        obj_path: 原始mesh模型的地址  
        volume: 体素模型，(32,32,32)  
        points(N个采样点): (N, 3)  
        sym_points(另外N个采样点): (N, 3)  
        plane(对称平面): (4)  
        quats(旋转对称轴):  (4)          
        输出三维视图的视频到name中  
    '''
    # 创建3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 画出原始模型
    if obj_path is not None:
        obj_path = os.path.join(obj_path, "model_normalized.obj")
        scene = trimesh.load(obj_path)
        for _, mesh in scene.geometry.items():
            # 提取顶点和面
            vertices = point2voxel(mesh.vertices)
            faces = mesh.faces
            # 绘制三角面片
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], linewidth=0.2, triangles=faces, alpha=0.7, edgecolor='gray')
    
    # 绘制体素
    if volume is not None:
        ax.voxels(volume, facecolors='cyan', edgecolors='k')
    
    # 画出一些点
    if points is not None:
        points = point2voxel(points.cpu())
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g')
    if sym_points is not None:
        sym_points = point2voxel(sym_points.cpu())
        ax.scatter(sym_points[:, 0], sym_points[:, 1], sym_points[:, 2], c='r')
    
    # 画出对称平面
    if planes is not None:
        for j, planej in enumerate(planes):
            planej = planej.cpu().numpy()[0]

            # # 定义平面和旋转轴
            # plane_normal = np.array([1, 0, 0])  # 平面的法向量 (x, y, z)
            # d = 5  # 平面与原点的距离
            # quaternion = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # 四元数表示的旋转 (这里是90度绕z轴旋转)

            # 定义平面上的点
            # xx, yy = np.meshgrid(range(int(voxel.shape[0]/2)-5,int(voxel.shape[0]/2+5)), range(int(voxel.shape[1]/2-5),int(voxel.shape[1]/2)+5))
            # zz = (-planej[..., 0] * xx - planej[..., 1] * yy - planej[..., 3]) * 1. / planej[..., 2]
            indices = np.argsort(np.abs(planej[:3]), axis=-1)

            xx, yy = np.meshgrid(range(int(gridSize/2-10),int(gridSize/2+10)), range(int(gridSize/2)-10,int(gridSize/2)+10))
            xx1 = voxel2point(xx)
            yy1 = voxel2point(yy)
            zz = (-planej[indices[0]] * xx1 - planej[indices[1]] * yy1 - planej[3]) * 1. / planej[indices[2]]
            zz = point2voxel(zz)
            # zz = np.round(np.clip(zz, a_min=0, a_max=gridSize-1))
            xyz = np.empty(shape=(3,)+xx.shape)
            xyz[indices[0]] = xx
            xyz[indices[1]] = yy
            xyz[indices[2]] = zz

            # 绘制平面
            if(j % 3 == 0):
                ax.plot_surface(xyz[0], xyz[1], xyz[2], alpha=0.7, color='lightblue')
            if(j % 3 == 1):
                ax.plot_surface(xyz[0], xyz[1], xyz[2], alpha=0.7, color='lightgreen')
            if(j % 3 == 2):
                ax.plot_surface(xyz[0], xyz[1], xyz[2], alpha=0.7, color='lightpink')
        
    if quats is not None:
        for j, quatj in enumerate(quats):
            # 生成旋转矩阵
            quatj = quatj.cpu().numpy()
            rotation = R.from_quat(quatj)
            # 旋转向量
            start = np.array([gridSize/2, gridSize/2, gridSize/2])
            axis = rotation.as_rotvec()[0]  # 获取旋转向量
            # end = start + axis  # 旋转后的终点
            # 绘制旋转轴
            ax.quiver(start[0], start[1], start[2], axis[0], axis[1], axis[2], color='m', length=np.linalg.norm(axis)*2)


    # 设置图像显示范围
    ax.set_xlim(0, gridSize)
    ax.set_ylim(0, gridSize)
    ax.set_zlim(0, gridSize)

    # 显示图像
    def update(num):
        ax.view_init(elev=num, azim=num)

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 10), interval=100)
    ani.save(save_path, writer='ffmpeg', fps=10)
    plt.close(fig)


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    

