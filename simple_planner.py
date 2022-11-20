import itertools
import random

import numpy as np
from numpy import unique, where
import os
import sys
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FILE = os.path.dirname(__file__)
if FILE not in sys.path:
    sys.path.append(FILE)

from algorithms import Fuzzy


class Planner:
    def __init__(self, vessel_list, sus_target_list):
        """
        :param env: the environment instance, so that the planner can reference environment vars
        """
        self.sus_target_list = sus_target_list
        self.agent_number = len(vessel_list)

    def plan(self, opt="Agg"):
        if opt not in ["Agg", "k_means", "MBKmeans", "c_fuzzy"]:
            print("Planner {} Algorithm error, choose a valid algorithm.".format(opt))
            return []
        print("use cluster: {}".format(opt))
        planner_name = "plan_" + opt
        planner = getattr(self, planner_name)
        assignment_list_temp = planner()
        return assignment_list_temp

    def plan_Agg(self):

        """
        :return: assignment_list: a list where entry i is an ordered list of the tasks (by number) assigned to robot i. e.g.
            [2,7,5]
        """
        cluster_scale = [1., 4.]
        if len(self.sus_target_list) == 0:
            return []
        elif len(self.sus_target_list) == 1:
            return [(0,)]
        task_state = [(sus_tar.center.x * cluster_scale[0], sus_tar.center.y * cluster_scale[1]) for sus_tar in self.sus_target_list]
        assignment_list_temp = list()
        X = task_state
        # 定义模型
        model = AgglomerativeClustering(n_clusters=min(self.agent_number, len(self.sus_target_list)), linkage='single')  # 'ward', 'complete', 'average', 'single'
        # 模型拟合与聚类预测
        yhat = model.fit_predict(X)
        # 检索唯一群集
        clusters = unique(yhat)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(yhat == cluster)
            # 创建这些样本的散布
            assignment_list_temp.append(row_ix[0])
        return assignment_list_temp

    def plan_k_means(self):
        if len(self.sus_target_list) == 0:
            return []
        elif len(self.sus_target_list) == 1:
            return [(0,)]
        task_state = [(sus_tar.center.x, sus_tar.center.y) for sus_tar in self.sus_target_list]
        assignment_list_temp = list()
        k_means = KMeans(init="k-means++", n_clusters=min(self.agent_number, len(self.sus_target_list)), n_init=10)
        yhat = k_means.fit(task_state)
        clusters = unique(yhat.labels_)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(yhat.labels_ == cluster)
            # 创建这些样本的散布
            assignment_list_temp.append(row_ix[0])
        return assignment_list_temp

    def plan_MBKmeans(self):
        if len(self.sus_target_list) == 0:
            return []
        elif len(self.sus_target_list) == 1:
            return [(0,)]
        task_state = [(sus_tar.center.x, sus_tar.center.y) for sus_tar in self.sus_target_list]
        assignment_list_temp = list()

        mbk = MiniBatchKMeans(
            init="k-means++",
            n_clusters=4,
            batch_size=256,
            n_init=10,
            max_no_improvement=10,
            verbose=0,
        )

        yhat = mbk.fit(task_state)
        clusters = unique(yhat.labels_)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(yhat.labels_ == cluster)
            # 创建这些样本的散布
            assignment_list_temp.append(row_ix[0])
        return assignment_list_temp

    def plan_c_fuzzy(self):
        if len(self.sus_target_list) == 0:
            return []
        elif len(self.sus_target_list) == 1:
            return [(0,)]

        task_state = [(sus_tar.center.x, sus_tar.center.y) for sus_tar in self.sus_target_list]
        assignment_list_temp = list()

        F = Fuzzy()
        data = np.array(task_state)
        xy_scale = (max(data[:, 0]) - min(data[:, 0])) / (max(data[:, 1]) - min(data[:, 1]))
        data[:, 0] = data[:, 0] / (xy_scale)
        res_U = F.fuzzy(data, self.agent_number, 2)

        l = [1] * self.agent_number
        idxs = (res_U == np.array(l)).nonzero()
        for i in range(self.agent_number):
            _idx = (idxs[1] == i).nonzero()
            assignment_list_temp.append(_idx[0])
        return assignment_list_temp

    def plan_test(self, point_list, mode="Agg"):
        assignment_list_temp = list()
        if mode == "Agg":
            X = point_list
            # 定义模型
            model = AgglomerativeClustering(n_clusters=min(self.agent_number, len(self.sus_target_list)), linkage='ward')  # 'ward', 'complete', 'average', 'single'
            # 模型拟合与聚类预测
            yhat = model.fit_predict(X)
            # 检索唯一群集
            clusters = unique(yhat)
            # 为每个群集的样本创建散点图
            for cluster in clusters:
                # 获取此群集的示例的行索引
                row_ix = where(yhat == cluster)
                # 创建这些样本的散布
                assignment_list_temp.append(row_ix[0])
        elif mode == "c_fuzzy":
            F = Fuzzy()
            data = np.array(point_list)
            res_U = F.fuzzy(data, self.agent_number, 2)

            l = [1] * self.agent_number
            idxs = (res_U == np.array(l)).nonzero()
            for i in range(self.agent_number):
                _idx = (idxs[1] == i).nonzero()
                assignment_list_temp.append(_idx[0])

        return assignment_list_temp


if __name__ == "__main__":
    a = [1, 2, 3, 4]
    color_bar = ['r', 'g', 'b', 'y', 'k']
    cluster_scale = [0.55, 0.17]
    planner = Planner(a, a)
    point_list = list()
    for _ in range(10):
        x = 121 + random.random() * cluster_scale[0]
        y = 22 + random.random() * cluster_scale[1]
        point_list.append([x, y])

    assign = planner.plan_test(point_list, mode='c_fuzzy')
    point_list = np.array(point_list)
    fig_x = 20
    fig_y = 20 / cluster_scale[0] * cluster_scale[1]
    plt.figure(figsize=(fig_x, fig_y))
    for i in range(len(assign)):
        _assign = np.array(list(assign[i]))
        xy = point_list[_assign]
        x = xy[:, 0]  # / cluster_scale[0]
        y = xy[:, 1]  # / cluster_scale[1]
        plt.scatter(x, y, color=color_bar[i])
    plt.show()
