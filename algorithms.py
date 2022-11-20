import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time


class Fuzzy:
    def __init__(self):
        self.MAX = 10000.
        self.Epsilon = 0.0000001

    def print_matrix(self, _list):
        """
        以可重复的方式打印矩阵
        """
        for i in range(0, len(_list)):
            print(_list[i])

    def initialize_U(self, data, cluster_number):
        """
        这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
        """
        MAX = self.MAX
        U = []
        for i in range(0, len(data)):
            current = []
            rand_sum = 0.0
            for j in range(0, cluster_number):
                dummy = random.randint(1, int(MAX))
                current.append(dummy)
                rand_sum += dummy
            for j in range(0, cluster_number):
                current[j] = current[j] / rand_sum
            U.append(current)
        return U

    def distance(self, point, center):
        """
        该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
        """
        if len(point) != len(center):
            return -1
        dummy = 0.0
        for i in range(0, len(point)):
            dummy += abs(point[i] - center[i]) ** 2
        return math.sqrt(dummy)

    def end_conditon(self, U, U_old):
        """
        结束条件。当U矩阵随着连续迭代停止变化时，触发结束
        """
        Epsilon = self.Epsilon
        for i in range(0, len(U)):
            for j in range(0, len(U[0])):
                if abs(U[i][j] - U_old[i][j]) > Epsilon:
                    return False
        return True

    def normalise_U(self, U):
        """
        在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
        """
        for i in range(0, len(U)):
            maximum = max(U[i])
            for j in range(0, len(U[0])):
                if U[i][j] != maximum:
                    U[i][j] = 0
                else:
                    U[i][j] = 1
        return U

    def fuzzy(self, data, cluster_number, m):
        """
        这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
        输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
        """
        # 初始化隶属度矩阵U
        U = self.initialize_U(data, cluster_number)
        # self.print_matrix(U)
        # 循环更新U
        while (True):
            # 创建它的副本，以检查结束条件
            U_old = copy.deepcopy(U)
            # 计算聚类中心
            C = []
            for j in range(0, cluster_number):
                current_cluster_center = []
                for i in range(0, len(data[0])):
                    dummy_sum_num = 0.0
                    dummy_sum_dum = 0.0
                    for k in range(0, len(data)):
                        # 分子
                        dummy_sum_num += (U[k][j] ** m) * data[k][i]
                        # 分母
                        dummy_sum_dum += (U[k][j] ** m)
                    # 第i列的聚类中心
                    current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
                # 第j簇的所有聚类中心
                C.append(current_cluster_center)

            # 创建一个距离向量, 用于计算U矩阵。
            distance_matrix = []
            for i in range(0, len(data)):
                current = []
                for j in range(0, cluster_number):
                    current.append(self.distance(data[i], C[j]))
                distance_matrix.append(current)

            # 更新U
            for j in range(0, cluster_number):
                for i in range(0, len(data)):
                    dummy = 0.0
                    for k in range(0, cluster_number):
                        # 分母
                        dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                    U[i][j] = 1 / dummy

            if self.end_conditon(U, U_old):
                print("已完成聚类")
                break

        U = self.normalise_U(U)
        return U


class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 50  # 迭代数目
        self.num = 200  # 粒子数目
        self.num_city = num_city  # 城市数
        self.location = data  # 城市的位置坐标
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        # 初始化所有粒子
        # self.particals = self.random_init(self.num, num_city)
        self.particals = self.greedy_init(self.dis_mat, num_total=self.num, num_city=num_city)
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 随机初始化
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算一条路径的长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1 < l2:
            return one, l1
        else:
            return one, l2

    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one, self.dis_mat)
        return one, l2

    # 迭代操作
    def pso(self):
        for cnt in range(1, self.iter_max):
            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                # 变异
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            # print(cnt, self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        # 画出最终路径
        return self.location[best_path], best_length


def load_data(input_path):
    with open(input_path, 'r', encoding="utf8") as f:
        task_ves_info = json.load(f)
        susTargetInfo = task_ves_info["content"]["arguments"]["susTargetInfo"]
        sustarget_list = list()
        for sus_tar_info in susTargetInfo:
            area = np.array(sus_tar_info["susTargetArea"])
            center = [min(area[:, 0]) + (max(area[:, 0]) - min(area[:, 0])) / 2,
                      min(area[:, 1]) + (max(area[:, 1]) - min(area[:, 1])) / 2]
            sustarget_list.append(center)
        return sustarget_list


if __name__ == '__main__':
    colorbar = ['r', 'b', 'g', 'c', 'y', 'k']
    input_path = "input/input.json"
    start = time.time()
    data = load_data(input_path)
    ves_pos = [[]]
    # 调用模糊C均值函数
    F = Fuzzy()
    data = np.array(data)[:, :2]
    xy_scale = (max(data[:, 0]) - min(data[:, 0])) / (max(data[:, 1]) - min(data[:, 1]))
    data[:, 0] = data[:, 0] / (xy_scale * 1)
    res_U = F.fuzzy(data, 4, 2)

    # print(data[:, 0])
    l = [1, 1, 1, 1]
    idxs = (res_U == np.array(l)).nonzero()
    # print(idxs[1])
    assignment_list = list()
    for i in range(4):
        _idx = (idxs[1] == i).nonzero()
        # print(_idx[0])
        plt.scatter(data[_idx, 0], data[_idx, 1], color=colorbar[i])
        assignment_list.append(_idx[0])

    for i in range(4):
        ass_data = data[assignment_list[i]]
        model = PSO(num_city=ass_data.shape[0], data=ass_data.copy())
        Best_path, Best = model.run()
        x = np.array(Best_path)[:, 0]
        y = np.array(Best_path)[:, 1]
        plt.plot(x, y, color=colorbar[i])

    # 计算
    print("用时：{0}".format(time.time() - start))
    plt.show()
