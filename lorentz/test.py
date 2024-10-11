import pickle
import random
import unittest

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lorentz.transfer import Lorentz, ndcg_at_k, ndcg_at_k_asc

import torch.nn as nn
import torch.optim as optim


def fit_regression(x, y, degree=2):
    # 定义模型
    poly = torch.nn.Linear(degree + 1, 1)

    # 生成多项式特征
    X = torch.stack([x ** i for i in range(degree + 1)]).T

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(poly.parameters(), lr=0.001)

    # 训练模型
    for _ in range(10000):
        optimizer.zero_grad()
        y_pred = poly(X)
        loss = criterion(y_pred, y.view(-1, 1))
        loss.backward()
        optimizer.step()

    return poly




# 函数：计算R²
def r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyTestCase(unittest.TestCase):

    def dtw_distance(self, seq1, seq2):
        # 初始化距离矩阵
        n, m = len(seq1), len(seq2)
        dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw_matrix[0][0] = 0

        # 计算每个点的DTW距离
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i - 1] - seq2[j - 1])
                # 取左，下，左下三个方向的最小值然后加上当前cost
                last_min = min(dtw_matrix[i - 1][j],  # 插入
                               dtw_matrix[i][j - 1],  # 删除
                               dtw_matrix[i - 1][j - 1])  # 匹配
                dtw_matrix[i][j] = cost + last_min

        return dtw_matrix[n][m]

    def generate_lists_violating_triangle_inequality(self, max_length=6, value_range=(-9, 9)):
        while True:
            # 随机生成三个长度和值都是随机的list
            #list1 = [random.randint(value_range[0], value_range[1]) for _ in range(random.randint(3, max_length))]
            list1 =[0,1,2,3]
            #list2 = [random.randint(value_range[0], value_range[1]) for _ in range(random.randint(3, max_length))]
            list2 =[3,2,1,0]
            #list3 = [random.randint(value_range[0], value_range[1]) for _ in range(random.randint(3, max_length))]
            list3 = [0,2,5]
            # 计算两两之间的DTW距离
            dist12 = self.dtw_distance(list1, list2)
            dist23 = self.dtw_distance(list2, list3)
            dist13 = self.dtw_distance(list1, list3)

            # 检查是否违反三角不等式
            # if dist12 + dist23 < dist13 or dist12 + dist13 < dist23 or dist13 + dist23 < dist12:
            #     return list1, list2, list3

            return list1, list2, list3
    def test_gen_dtw_inequality(self):
        a, b, c = self.generate_lists_violating_triangle_inequality(3, value_range=(0, 4))
        print(a, b, c)
        print(self.dtw_distance(a, b), self.dtw_distance(b, c), self.dtw_distance(a, c))

    def triangle_inequality(self, a, b, c):
        max_side = max(a, b, c)
        sum_other_two = a + b + c - max_side
        if sum_other_two >= max_side:
            return False, 0.0, 0.0
        else:
            shortfall = max_side - sum_other_two
            return True, shortfall, shortfall / max_side

    def eps_statistic(self, dataset, simtype):
        smallest_values = []
        for row in dataset:
            # Sort the row and pick the first 100 values
            smallest_values.extend(np.sort(row)[:100])
            # Calculate metrics
        smallest_values = np.array(smallest_values)
        mean = np.mean(smallest_values)

        data = dataset.reshape(-1)
        min_val = np.min(data)
        max_val = np.max(data)
        values0 = data[(data == min_val)]
        values0_percentage = len(values0) / len(data) * 100
        values1 = data[(data == max_val)]
        values1_percentage = len(values1) / len(data) * 100
        print(
            f"{simtype}-all:{100 - values0_percentage - values1_percentage}|{values0_percentage}|{values1_percentage}")
        data = smallest_values.reshape(-1)
        values0 = data[(data == min_val)]
        values0_percentage = len(values0) / len(data) * 100
        values1 = data[(data == max_val)]
        values1_percentage = len(values1) / len(data) * 100
        print(
            f"{simtype}-100:{100 - values0_percentage - values1_percentage}|{values0_percentage}|{values1_percentage}")

    def test_sim_distribution(self):
        analysis_results = []
        simtype = 'lcss100'
        simtypes = ['edr10', 'edr20', 'edr50', 'edr100', 'edr200', 'edr500', 'edr1000']
        simtypes = ['lcss10', 'lcss20', 'lcss50', 'lcss100', 'lcss200', 'lcss500', 'lcss1000']
        city = 'chengdu'
        for simtype in simtypes:
            distancepath = f'../data_set/{city}/remote/{simtype}_10000x10000.pkl'
            # distance_matrix = pickle.load(open(distancepath, 'rb'))
            distance_matrix = np.array(pickle.load(open(distancepath, 'rb')))
            smallest_values = []
            for row in distance_matrix:
                # Sort the row and pick the first 100 values
                smallest_values.extend(np.sort(row)[:100])
                # Calculate metrics
            smallest_values = np.array(smallest_values)
            mean = np.mean(smallest_values)

            data = distance_matrix.reshape(-1)
            min_val = np.min(data)
            max_val = np.max(data)
            values0 = data[(data == min_val)]
            values0_percentage = len(values0) / len(data) * 100
            values1 = data[(data == max_val)]
            values1_percentage = len(values1) / len(data) * 100
            print(
                f"{simtype}-all:{100 - values0_percentage - values1_percentage}|{values0_percentage}|{values1_percentage}")
            data = smallest_values.reshape(-1)
            values0 = data[(data == min_val)]
            values0_percentage = len(values0) / len(data) * 100
            values1 = data[(data == max_val)]
            values1_percentage = len(values1) / len(data) * 100
            print(
                f"{simtype}-100:{100 - values0_percentage - values1_percentage}|{values0_percentage}|{values1_percentage}")

    def test_dist(self):
        simtype = 'dtw'
        city = 'porto'
        portopath = f'../data_set/{city}/{simtype}_10000x10000.pkl'
        with open(portopath, 'rb') as file:
            porto_dtw = pickle.load(file)
        city = 'chengdu'
        chengdupath = f'../data_set/{city}/{simtype}_10000x10000.pkl'
        with open(chengdupath, 'rb') as file:
            chengdu_dtw = pickle.load(file)
        for i in range(10000):
            for j in range(10000):
                if i == j:
                    continue
                self.assertNotEqual(porto_dtw[i][j], chengdu_dtw[i][j])

    def test_triangle_inequality(self):
        # chengdu frechet   total_500000|cnt291|div0.02688|norm_div0.4311
        # chengdu hausdorff total_500000|cnt0|div0.0|norm_div0.0
        # chengdu dtw       total_500000|cnt98575|div3737094449.9959445|norm_div11202.606175646817
        # chengdu erp       total_500000|cnt0|div0.0|norm_div0.0

        # porto frechet     total_500000|cnt2511|div17.27347564260093|norm_div155.99747887841525
        # porto hausdorff   total_500000|cnt0|div0.0|norm_div0.0
        # porto dtw         total_500000|cnt101762|div16110201816.254496|norm_div9725.85444127995
        # porto erp         total_500000|cnt0|div0.0|norm_div0.0
        simtype = 'sspd'
        city = 'geolife'
        set_seed(2024)
        for simtype in ['dtw', 'edr20', 'edr50', 'edr100', 'edr200', 'edr400', 'edr750', 'sspd', ]:
            distancepath = f'../data_set/{city}/remote/{simtype}_10000x10000.pkl'
            with open(distancepath, 'rb') as file:
                distance_matrix = pickle.load(file)
            distance_matrix = np.array(distance_matrix)
            total, cnt, sum_div, sum_norm_div = 0, 0, 0.0, 0.0
            for _ in range(1000000):
                i, j, k = random.sample(range(10000), 3)
                a, b, c = distance_matrix[i][j], distance_matrix[i][k], distance_matrix[j][k]
                total += 1
                flag, div, norm_div = self.triangle_inequality(a, b, c)
                if flag:
                    cnt += 1
                    sum_div += div
                    sum_norm_div += norm_div
            print(f'{simtype}:total_{total}|cnt{cnt}|div{sum_div}|norm_div{sum_norm_div}')
            self.eps_statistic(distance_matrix, simtype)
            # tq.set_postfix_str(f'total_{total}|cnt{cnt}|div{sum_div}|norm_div{sum_norm_div}')

    def test_draw2(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # Setting up the x range
        x = np.linspace(1, 4, 400)
        # Calculating y from the hyperbola equation y^2 - x^2 + 1 = 0
        y_positive = np.sqrt(x ** 2 - 1)
        y_negative = -np.sqrt(x ** 2 - 1)

        # Points
        points = {'A': (1, 0), 'B': (2, -np.sqrt(3)), 'C': (3, 2 * np.sqrt(2))}
        px, py = zip(*points.values())

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plotting the hyperbola
        plt.plot(x, y_positive, 'b')
        plt.plot(x, y_negative, 'b')

        # Plotting the points
        plt.scatter(px, py, color='red')
        for label, (px, py) in points.items():
            plt.annotate(label, (px, py), textcoords="offset points", xytext=(5, -5), ha='center')

        # Connecting points with orange lines
        plt.plot([points['A'][0], points['B'][0]], [points['A'][1], points['B'][1]], 'orange')
        plt.plot([points['B'][0], points['C'][0]], [points['B'][1], points['C'][1]], 'orange')
        plt.plot([points['C'][0], points['A'][0]], [points['C'][1], points['A'][1]], 'orange')

        # Setting the aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')

        # Labels and title
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        # plt.title('Right branch of the hyperbola y^2 - x^2 + 1 = 0 with triangle ABC')
        plt.legend()

        plt.grid(True)
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(0, 5)
        plt.ylim(-3, 4)
        plt.show()

    def test_cuda(self):
        import torch
        a = torch.ones((5, 10), dtype=torch.float64, device='cuda')
        b = nn.Linear(10, 1, device='cuda', dtype=torch.float64)
        c = b(a)
        print(c)

    def read_traj(self, file_path):
        columns = ['Latitude', 'Longitude', '0', 'Altitude', 'NumDays', 'Date', 'Time']

        # 读取 .plt 文件，跳过前 6 行（文件头部信息）
        df = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
        traj = df[['Longitude', 'Latitude']].values.tolist()
        return traj

    def chk_range(self, traj):
        for point in traj:
            if point[0] < 115 or point[0] >= 118:
                return False
            if point[1] < 39 or point[1] >= 42:
                return False
        return True

    def rmv_same_point(self, traj):
        lastlon, lastlat = 0, 0
        new_traj = []
        for point in traj:
            lon, lat = point
            if lastlon == lon and lastlat == lat:
                continue
            else:
                new_traj.append(point)
                lastlon, lastlat = lon, lat
        return new_traj

    def test_construct_geolife(self):
        traj_list = []
        root_path = '../data_set/geolife/Data'
        # root_folder = "/path/to/your/folder"
        import os, pickle
        # 遍历主文件夹
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)

            # 检查是否是文件夹
            if os.path.isdir(folder_path):
                trajectory_folder = os.path.join(folder_path, "Trajectory")

                # 检查是否存在trajectory文件夹
                if os.path.exists(trajectory_folder) and os.path.isdir(trajectory_folder):

                    # 遍历trajectory文件夹下的plt文件
                    for plt_filename in os.listdir(trajectory_folder):
                        if plt_filename.endswith(".plt"):
                            plt_filepath = os.path.join(trajectory_folder, plt_filename)

                            # 使用pickle读取plt文件
                            data = self.read_traj(plt_filepath)
                            traj_list.append(data)
                            # 在这里你可以处理读取到的数据，比如打印或进行其他操作
                            # print(f"Read data from {plt_filepath}")
        print(len(traj_list))
        new_list = []
        for traj in traj_list:
            if self.chk_range(traj):
                new_list.append(traj)
        print(len(new_list))
        valid_list = []

        for traj in new_list:
            new_traj = self.rmv_same_point(traj)
            # new_traj = self.clear_out_range(new_traj)
            if len(new_traj) > 20:
                valid_list.append(new_traj)
                # len_cnt.append(len(new_traj))
        # npl = np.array(len_cnt)
        # print(np.sum(npl <= 500))


        sorted_list = sorted(valid_list, key=len)[1000:11000]
        for i, traj in enumerate(sorted_list):
            sorted_list[i] = sorted_list[i][:100]
        random.shuffle(sorted_list)
        a = len(sorted_list)
        with open(f"../data_set/geolife/trajs_{a}.plt", "wb") as file:
            pickle.dump(sorted_list, file)

    def test_mode_geolife(self):
        with open('../data_set/geolife/trajs_10000.plt', 'rb') as file:
            traj_list = pickle.load(file)
        for i, traj in enumerate(traj_list):
            traj_list[i] = traj_list[i][:100]
        with open(f"../data_set/geolife/trajs_10000.pkl", "wb") as file:
            pickle.dump(traj_list, file)

    def test_draw(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Setting up the x and y ranges
        x = np.linspace(-1, 1, 400)
        y = np.linspace(-1, 1, 400)
        x, y = np.meshgrid(x, y)

        # Calculating z from the hyperboloid equation x^2 + y^2 - z^2 + 1 = 0
        # We only take the positive root for the upper half
        z_hyperboloid = np.sqrt(x ** 2 + y ** 2 + 1)

        # z for the auxiliary plane
        z_plane = np.zeros(x.shape)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Hyperboloid
        ax.plot_surface(x, y, z_hyperboloid, color='b', alpha=0.7, rstride=20, cstride=20)
        # Auxiliary plane at z=0
        ax.plot_surface(x, y, z_plane, color='lightgrey', alpha=0.5)

        # Removing the grid
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Setting the color of the panes to white
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # # Setting the axes at the origin
        # ax.xaxis._axinfo['juggled'] = (0, 0, 0)
        # ax.yaxis._axinfo['juggled'] = (1, 1, 1)
        # ax.zaxis._axinfo['juggled'] = (2, 2, 2)

        # Labels and title
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Upper half of a two-sheet hyperboloid (x^2 + y^2 - z^2 + 1 = 0) with auxiliary plane at z=0')

        # Setting the viewing angle (elevation, azimuth)
        ax.view_init(elev=15, azim=240)

        plt.show()

    def test_ndcg(self):
        a = np.array([[1, 2, 3, 4, 6, 5, 7]])
        b = np.array([[1, 2, 3, 4, 5, 6, 7]])
        c = ndcg_at_k(a, b, 3)
        print(c)
        a = np.array([[1, 2, 3, 4, 5, 6, 7]])
        b = np.array([[1, 3, 2, 4, 5, 6, 7]])
        c = ndcg_at_k_asc(a, b, 3)
        print(c)

    def test_something(self):
        l = Lorentz()
        a = torch.randn((10, 225))
        b = torch.randn((10, 225))

        ld = l._simple_lorentz_dist(a, b)
        nd = l._normal_dist(a, b)
        # print(f"dist :{nd} \nlorentz dist:{ld}")
        sorted_nd, indices = torch.sort(nd)
        sorted_ld = ld[indices]
        print(f"dist :{sorted_nd} \nlorentz dist:{sorted_ld}")


if __name__ == '__main__':
    unittest.main()
