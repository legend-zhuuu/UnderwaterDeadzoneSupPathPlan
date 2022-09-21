import itertools

import numpy as np
from numpy import unique, where
from sklearn.cluster import KMeans, AgglomerativeClustering
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

        task_state = [(sus_tar.center.x, sus_tar.center.y) for sus_tar in self.sus_target_list]
        assignment_list_temp = list()
        X = task_state
        # 定义模型
        model = AgglomerativeClustering(n_clusters=self.agent_number)
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
            # plt.scatter(X[row_ix, 0], X[row_ix, 1])
        # 绘制散点图
        # plt.show()

        return assignment_list_temp




