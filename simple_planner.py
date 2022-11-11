import itertools
import random

import numpy as np
from numpy import unique, where
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Planner:

    def __init__(self, vessel_list, sus_target_list):
        """
        :param env: the environment instance, so that the planner can reference environment vars
        """
        self.sus_target_list = sus_target_list
        self.agent_number = len(vessel_list)

    def plan(self):

        """
        :return: assignment_list: a list where entry i is an ordered list of the tasks (by number) assigned to robot i. e.g.
            [2,7,5]
        """
        cluster_scale = [1.2, 1.]
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

    def plan_test(self, point_list):
        assignment_list_temp = list()
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
        return assignment_list_temp


if __name__ == "__main__":
    a = [1, 2, 3, 4]
    color_bar = ['r', 'g', 'b', 'y', 'k']
    cluster_scale = [1., 4.]
    planner = Planner(a, a)
    point_list = list()
    for _ in range(10):
        x = random.random() * cluster_scale[0]
        y = random.random() * cluster_scale[1]
        point_list.append([x, y])

    assign = planner.plan_test(point_list)
    point_list = np.array(point_list)
    for i in range(len(assign)):
        _assign = np.array(list(assign[i]))
        xy = point_list[_assign]
        x = xy[:, 0] #/ cluster_scale[0]
        y = xy[:, 1] #/ cluster_scale[1]
        plt.scatter(x, y, color=color_bar[i])
    plt.show()



