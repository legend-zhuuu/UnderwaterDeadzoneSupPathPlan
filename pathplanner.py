import numpy as np
import math
import argparse
import scipy.io as sio
from datetime import datetime
import os
import sys
import importlib
import time
import json
from copy import deepcopy
from itertools import permutations
from geographiclib.geodesic import Geodesic

FILE = os.path.dirname(__file__)
if FILE not in sys.path:
    sys.path.append(FILE)

from simple_planner import Planner
from model import Point, Vessel, SusTarget, Target, Area


class PathPlanner:
    def __init__(self, task_ves_info):
        # 导入json文件中的任务和舰艇信息
        self.task_ves_info = task_ves_info

        # information
        self.target_number = len(self.task_ves_info["content"]["arguments"]["susTargetInfo"])
        self.agent_number = len(self.task_ves_info["content"]["arguments"]["vesInfo"])

        self.target_threat_radius = self.task_ves_info["content"]["arguments"]["config"]["targetThreatRadius"]
        self.dead_zone_width = self.task_ves_info["content"]["arguments"]["config"]["deadZoneWidth"]
        self.start_point_dis = self.task_ves_info["content"]["arguments"]["config"]["startPointdis"]
        self.search_spd = self.task_ves_info["content"]["arguments"]["config"]["speed"]
        self.sonar_length = self.task_ves_info["content"]["arguments"]["config"]["sonarLength"]
        self.sonar_length_plus = 5
        self.target_threat_radius_plus = 20
        self.initial = self.task_ves_info["content"]["arguments"]["initial"]

        # system arguments
        self.work_state = True
        self.system_state = {
            "workState": True,
            "inputState": 0,
            "outputState": True,
            "msg": "ok"
        }

        self.jqtime = 0

        self.target_list = list()
        self.sustarget_list = list()
        self.vessel_list = list()

        # load information
        self.load_global_info()
        self.load_targets_pos_info()
        self.load_sus_target_info()
        self.load_ves_info()

        # geographiclib
        self.geod = Geodesic.WGS84
        self.geod = Geodesic(6378388, 1 / 297.0)

    def load_global_info(self):
        self.task_area = Area(self.task_ves_info["content"]["arguments"]["taskArea"])

    def load_targets_pos_info(self):
        self.targetInfo = self.task_ves_info["content"]["arguments"]["targetInfo"]
        for tar_info in self.targetInfo:
            target = Target(tar_info["targetId"], tar_info["targetPos"])
            self.target_list.append(target)

    def load_sus_target_info(self):
        self.susTargetInfo = self.task_ves_info["content"]["arguments"]["susTargetInfo"]
        for sus_tar_info in self.susTargetInfo:
            sus_target = SusTarget(sus_tar_info["susTargetId"], sus_tar_info["susTargetArea"])
            self.sustarget_list.append(sus_target)
        if not self.input_valid(self.sustarget_list):
            self.system_state["inputState"] = 1
            print("input_error!")
            self.work_state = False

    def load_ves_info(self):
        self.vesInfo = self.task_ves_info["content"]["arguments"]["vesInfo"]
        for ves_info in self.vesInfo:
            vessel = Vessel(ves_info["tid"], ves_info["vesPos"], ves_info["sonarWidth"], ves_info["speed"])
            self.vessel_list.append(vessel)

    def input_valid(self, sus_target_list):
        sus_target_list_copy = sus_target_list.copy()
        while sus_target_list_copy:
            _sus_target = sus_target_list_copy.pop()
            _center = _sus_target.center
            for sus_target in sus_target_list_copy:
                center = sus_target.center
                if np.abs(center.x - _center.x) < (_sus_target.length / 2 + sus_target.length / 2) \
                        and np.abs(center.y - _center.y) < (_sus_target.width / 2 + sus_target.width / 2):
                    return False
        return True

    def empty_ves_dict(self):
        # todo
        pass

    @staticmethod
    def set_seed(seed_value):
        if seed_value == None:
            seed_val = np.random.randint(0, 1000)
        else:
            seed_val = seed_value
        np.random.seed(seed_val)
        return seed_val

    def get_args(self):
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('--planner', default='simple_planner', dest='planner', type=str,
                            help="planner to use: can be simple_planner, from_file, etc.?")
        parser.add_argument('--controller', default='simple_controller', dest='controller', type=str,
                            help="controller to use: can be simple_controller, FSMcontroller")
        parser.add_argument('--seed', default=None, dest='seed_val', type=int,
                            help="seed value for random number generator, int")
        parser.add_argument('--filename', default=None, dest='filename_str', type=str,
                            help="if importing from file, specify which file. Otherwise, choose the most recent matlab_out")
        parser.add_argument('--params', default='dependency_test_params', dest='params_name', type=str,
                            help='set the parameter file to be used')
        parser.add_argument('--n_tasks', default=self.target_number, dest='n_tasks', type=int,
                            help='number of tasks to use in simulation')
        parser.add_argument('--n_agents', default=self.agent_number, dest='n_agents', type=int,
                            help='number of agents to use in simulation')
        parser.add_argument('--n_dependencies', default=2, dest='n_dependencies', type=int,
                            help='number of dependencies between tasks')

        parser.add_argument('--save', default=False, dest='save', action='store_true',
                            help='save the data to the csv file')

        return parser.parse_args(args=[])

    @staticmethod
    def cross(p1, p2, p3):  # 跨立实验
        x1 = p2.x - p1.x
        y1 = p2.y - p1.y
        x2 = p3.x - p1.x
        y2 = p3.y - p1.y
        return x1 * y2 - x2 * y1

    def is_intersec(self, p1, p2, p3, p4):  # 判断两线段是否相交
        # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
        if (max(p1.x, p2.x) >= min(p3.x, p4.x)  # 矩形1最右端大于矩形2最左端
                and max(p3.x, p4.x) >= min(p1.x, p2.x)  # 矩形2最右端大于矩形最左端
                and max(p1.y, p2.y) >= min(p3.y, p4.y)  # 矩形1最高端大于矩形最低端
                and max(p3.y, p4.y) >= min(p1.y, p2.y)):  # 矩形2最高端大于矩形最低端

            # 若通过快速排斥则进行跨立实验 允许两个线段的顶点重合
            if (self.cross(p1, p2, p3) * self.cross(p1, p2, p4) < 0
                    and self.cross(p3, p4, p1) * self.cross(p3, p4, p2) < 0):
                is_cross = 1
            else:
                is_cross = 0
        else:
            is_cross = 0
        return is_cross

    @staticmethod
    def point_in_area(point, area):
        if area.ld_angle.x < point.x < area.rd_angle.x and area.ld_angle.y < point.y < area.lu_angle.y:
            return True
        return False

    @staticmethod
    def point_in_circle(point, center, radius):
        if np.linalg.norm([point.x - center.x, point.y - center.y]) <= radius:
            return True
        return False

    def comp_dist(self, l1, l2, point):
        return self.cross(l1, l2, point) / np.linalg.norm([l1.x - l2.x, l1.y - l2.y])

    @staticmethod
    def is_obtuse(A, B, C):
        v1 = np.array([B.x - A.x, B.y - A.y])
        v2 = np.array([C.x - A.x, C.y - A.y])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos < 0:
            return True
        return False

    def across_sus_area(self, start_point, end_point, area):
        if self.point_in_area(start_point, area) or self.point_in_area(end_point, area):
            return True
        if self.is_intersec(area.ld_angle, area.ru_angle, start_point, end_point) or \
                self.is_intersec(area.lu_angle, area.rd_angle, start_point, end_point):
            return True
        return False

    def pass_through_sus_tar(self, start_point, end_point):
        for sus_tar in self.sustarget_list:
            if self.point_in_area(start_point, sus_tar) or self.point_in_area(end_point, sus_tar):
                return sus_tar, True
            if self.is_intersec(sus_tar.ld_angle, sus_tar.ru_angle, start_point, end_point) or \
                    self.is_intersec(sus_tar.lu_angle, sus_tar.rd_angle, start_point, end_point):
                return sus_tar, True
            return [], False

    def pass_through_obs(self, start_point, end_point):
        obs_flag = False
        geo = self.geod.Direct(start_point.x - 90, start_point.y, 0, self.target_threat_radius)
        radius = geo["lat2"] + 90 - start_point.x
        for tar in self.target_list:
            center = tar.pos
            if self.point_in_circle(start_point, center=center, radius=radius) or \
                    self.point_in_circle(end_point, center, radius):
                return tar, True
            else:
                dist = abs(self.comp_dist(start_point, end_point, center))
                if dist >= radius:
                    continue
                elif dist == 0:
                    if (start_point.x <= center.x <= end_point.x or start_point.x >= center.x >= end_point.x) and \
                            (start_point.y <= center.y <= end_point.y or start_point.y >= center.y >= end_point.y):
                        return tar, True
                    else:
                        continue
                else:
                    if self.is_obtuse(start_point, end_point, center) or self.is_obtuse(end_point, start_point, center):
                        continue
                    else:
                        return tar, True
        return [], False

    def out_map(self, point1, point2):
        if point1.x < self.task_area.ld_angle.x or point1.x > self.task_area.ru_angle.x or \
                point1.y < self.task_area.ld_angle.y or point1.y > self.task_area.ru_angle.y or \
                point2.x < self.task_area.ld_angle.x or point2.x > self.task_area.ru_angle.x or \
                point2.y < self.task_area.ld_angle.y or point2.y > self.task_area.ru_angle.y:
            return True
        return False

    def candi_is_legal(self, candi_pair):
        p1, p2 = candi_pair
        _, obs_flag = self.pass_through_obs(p1, p2)
        out_map_flag = self.out_map(p1, p2)
        if not obs_flag and not out_map_flag:
            return True
        return False

    def go_to_other_angle(self, point, area):
        if area.type == "west_east":
            if point.y < area.ld_angle.y:
                if not self.across_sus_area(point, area.lu_angle, area):
                    return area.lu_angle
                else:
                    return area.ru_angle
            else:
                if not self.across_sus_area(point, area.ld_angle, area):
                    return area.ld_angle
                else:
                    return area.rd_angle
        else:
            if point.x < area.ld_angle.x:
                if not self.across_sus_area(point, area.rd_angle, area):
                    return area.rd_angle
                else:
                    return area.ru_angle
            else:
                if not self.across_sus_area(point, area.ld_angle, area):
                    return area.ld_angle
                else:
                    return area.lu_angle

    def insert_path_point(self, start_point, end_point, obs):
        insert_obs_list = list()
        geo1 = self.geod.Direct(obs.pos.x - 90, obs.pos.y, -45,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo2 = self.geod.Direct(obs.pos.x - 90, obs.pos.y, 45,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo3 = self.geod.Direct(obs.pos.x - 90, obs.pos.y, -135,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo4 = self.geod.Direct(obs.pos.x - 90, obs.pos.y, 135,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        angle_point_rd = Point([geo1["lat2"] + 90, geo1["lon2"]])
        angle_point_ru = Point([geo2["lat2"] + 90, geo2["lon2"]])
        angle_point_ld = Point([geo3["lat2"] + 90, geo3["lon2"]])
        angle_point_lu = Point([geo4["lat2"] + 90, geo4["lon2"]])
        angle_point_list = [angle_point_rd, angle_point_ru, angle_point_ld, angle_point_lu]

        for angle_point in angle_point_list:
            if (not self.is_intersec(start_point, angle_point, angle_point_list[0], angle_point_list[3])) and \
                    (not self.is_intersec(start_point, angle_point, angle_point_list[1], angle_point_list[2])) and \
                    (not self.is_intersec(angle_point, end_point, angle_point_list[0], angle_point_list[3])) and \
                    (not self.is_intersec(angle_point, end_point, angle_point_list[1], angle_point_list[2])):
                return [angle_point]
        else:
            if abs(end_point.x - start_point.x) < abs(end_point.y - start_point.y):
                if obs.pos.x < start_point.x:
                    if start_point.y < end_point.y:
                        insert_obs_list.append(angle_point_rd)
                        insert_obs_list.append(angle_point_ru)
                    else:
                        insert_obs_list.append(angle_point_ru)
                        insert_obs_list.append(angle_point_rd)
                else:
                    if start_point.y < end_point.y:
                        insert_obs_list.append(angle_point_ld)
                        insert_obs_list.append(angle_point_lu)
                    else:
                        insert_obs_list.append(angle_point_lu)
                        insert_obs_list.append(angle_point_ld)
            else:
                if obs.pos.y < start_point.y:
                    if start_point.x < end_point.x:
                        insert_obs_list.append(angle_point_lu)
                        insert_obs_list.append(angle_point_ru)
                    else:
                        insert_obs_list.append(angle_point_ru)
                        insert_obs_list.append(angle_point_lu)
                else:
                    if start_point.x < end_point.x:
                        insert_obs_list.append(angle_point_ld)
                        insert_obs_list.append(angle_point_rd)
                    else:
                        insert_obs_list.append(angle_point_rd)
                        insert_obs_list.append(angle_point_ld)
        return insert_obs_list

    def find_next_point(self, area, prev_point, next_area):
        out_point_list = list()
        candi_point_list = list()
        # 将搜索区域的四个顶点向外膨胀危险范围后的点作为预选的起点
        if next_area.type == 'west_east':
            # 5  6     7  8
            #    -------
            #    |     |
            #    -------
            # 1  2     3  4
            # 沿着东西方向的航路
            x, y = next_area.ld_angle.x, next_area.ld_angle.y
            # Direct中lattitude的取值范围为0-90，因此-90防止越界
            length_geo = self.geod.Direct(x - 90, y, 90, self.dead_zone_width / 2)
            sonar_length_geo = self.geod.Direct(x - 90, y, 0, self.sonar_length)
            length = length_geo["lon2"] - y
            sonar_length = sonar_length_geo["lat2"] + 90 - x
            p1 = next_area.ld_angle - Point([sonar_length, length])
            p2 = next_area.ld_angle - Point([0, length])
            p3 = next_area.rd_angle - Point([0, length])
            p4 = next_area.rd_angle - Point([-sonar_length, length])
            p5 = next_area.lu_angle - Point([sonar_length, -length])
            p6 = next_area.lu_angle - Point([0, -length])
            p7 = next_area.ru_angle - Point([0, -length])
            p8 = next_area.ru_angle - Point([-sonar_length, -length])
            _candi_point_list = [[p2, p4], [p3, p1], [p6, p8], [p7, p5]]
            for _candi_pair in _candi_point_list:
                if self.candi_is_legal(_candi_pair):
                    candi_point_list.append(_candi_pair)
        else:
            # 沿着南北方向的航路
            # 1           5
            # 2  -------  6
            #    |     |
            # 3  -------  7
            # 4           8
            x, y = next_area.ld_angle.x, next_area.ld_angle.y
            width_geo = self.geod.Direct(x - 90, y, 0, self.dead_zone_width / 2)
            sonar_length_geo = self.geod.Direct(x - 90, y, 90, self.sonar_length)
            width = width_geo["lat2"] + 90 - x
            sonar_length = sonar_length_geo["lon2"] - y
            p1 = next_area.lu_angle - Point([width, -sonar_length])
            p2 = next_area.lu_angle - Point([width, 0])
            p3 = next_area.ld_angle - Point([width, 0])
            p4 = next_area.ld_angle - Point([width, sonar_length])
            p5 = next_area.ru_angle - Point([-width, -sonar_length])
            p6 = next_area.ru_angle - Point([-width, 0])
            p7 = next_area.rd_angle - Point([-width, 0])
            p8 = next_area.rd_angle - Point([-width, sonar_length])
            _candi_point_list = [[p2, p4], [p3, p1], [p6, p8], [p7, p5]]
            for _candi_pair in _candi_point_list:
                if self.candi_is_legal(_candi_pair):
                    candi_point_list.append(_candi_pair)

        if len(candi_point_list) == 0:
            self.empty_ves_dict()
            print("Path plan error! All paths will hit target!")
        else:
            perform_list = list()
            out_flag = False
            for candi in candi_point_list:
                # candi = point1 + point2
                obs, obs_flag = self.pass_through_obs(prev_point, candi[0])

                area_flag = self.across_sus_area(prev_point, candi[0], area)
                next_area_flag = self.across_sus_area(prev_point, candi[0], next_area)
                dist = np.linalg.norm([prev_point.x - candi[0].x, prev_point.y - candi[0].y])
                perform_list.append([candi, [obs_flag, area_flag, next_area_flag].count(False), dist])
                if not any([obs_flag, area_flag, next_area_flag]):
                    out_flag = True
            else:
                if out_flag:
                    perform_list = [p for p in perform_list if p[1] == 3]
                    perform_list.sort(key=lambda x: x[2])
                    candi = perform_list[0][0]
                    out_point_list = out_point_list + candi
                    return out_point_list
                perform_list.sort(key=lambda x: x[1], reverse=True)
                candi = perform_list[0][0]
                if self.across_sus_area(prev_point, candi[0], area):
                    insert_angle_point = self.go_to_other_angle(prev_point, area)
                    out_point_list.append(insert_angle_point)
                    new_point = insert_angle_point
                    obs, obs_flag = self.pass_through_obs(new_point, candi[0])
                    if obs_flag:
                        insert_obs_point_list = self.insert_path_point(new_point, candi[0],
                                                                       obs)  # todo: 增加对插入路径点后的路径重新判断
                        out_point_list = out_point_list + insert_obs_point_list
                        new_point = insert_obs_point_list[-1]
                    if self.across_sus_area(new_point, candi[0], next_area):
                        insert_angle_point = self.go_to_other_angle(new_point, next_area)
                        out_point_list.append(insert_angle_point)
                        new_point = insert_angle_point
                else:
                    obs, obs_flag = self.pass_through_obs(prev_point, candi[0])
                    new_point = prev_point
                    if obs_flag:
                        insert_obs_point_list = self.insert_path_point(new_point, candi[0], obs)
                        out_point_list = out_point_list + insert_obs_point_list
                        new_point = insert_obs_point_list[-1]
                    if self.across_sus_area(new_point, candi[0], next_area):
                        insert_angle_point = self.go_to_other_angle(new_point, next_area)
                        out_point_list.append(insert_angle_point)
                        new_point = insert_angle_point
                out_point_list = out_point_list + candi
        return out_point_list

    def remake_assignment_list(self, assignment_list):
        sorted_assignment_list = list()
        # print("old", assignment_list)
        for assignment in assignment_list:
            y_list = [self.sustarget_list[int(x)].center.y for x in assignment]
            y_list, assignment = zip(*sorted(zip(y_list, assignment)))
            sorted_assignment_list.append(assignment)
        # print("sorted", sorted_assignment_list)

        # 四个船路径不交错
        # 枚举
        new_assignment_list = list()
        # four vessel position
        ves_start_point = [ves.pos for ves in self.vessel_list]

        for p in permutations(sorted_assignment_list):
            _ves_start_point = ves_start_point.copy()
            p = list(p)
            # four first point of task areas
            _task_start_point = [self.sustarget_list[int(assignment[0])].center for assignment in p]
            is_cross = False
            while len(_ves_start_point) > 1 and not is_cross:
                _point = _ves_start_point.pop()
                _end_point = _task_start_point.pop()
                for index in range(len(_ves_start_point)):
                    _other_point = _ves_start_point[index]
                    _other_end_point = _task_start_point[index]
                    if self.is_intersec(_point, _end_point, _other_point, _other_end_point):
                        is_cross = True
                        break
            if not is_cross:
                return p
        else:
            print("four paths may cross!")
            return sorted_assignment_list

    def output_json(self, assignment_list, system_state):
        assignment_list = self.remake_assignment_list(assignment_list)
        # print(assignment_list)

        # output
        ves_dict = dict()
        ves_dict.update({"id": 1})
        ves_dict.update({"method": "notice-event"})
        output = list()
        for i in range(len(assignment_list)):
            time_cost = 0
            ves = self.vessel_list[i]
            ves_id = ves.tid
            ves_output_info = dict()
            ves_output_info.update({"tid": ves_id})

            path_point_list = list()
            ves_start_pos = ves.pos

            task_points = assignment_list[i]
            prev_point = ves_start_pos

            first_area_info = self.susTargetInfo[int(task_points[0])]
            void_area = SusTarget("0000", [
                [0., 0.1],
                [0., 0.],
                [0.1, 0.],
                [0.1, 0.1],
                [0., 0.1]
            ])
            first_area = SusTarget(first_area_info["susTargetId"], first_area_info["susTargetArea"])

            out_path = self.find_next_point(void_area, prev_point, first_area)

            for index in range(len(out_path)):
                point = out_path[index]
                dist = self.geod.Inverse(point.x - 90, point.y, prev_point.x - 90, prev_point.y)["s12"]
                if index == 0:
                    vec = point - prev_point
                    angle = math.atan2(vec.y, vec.x) * 180 / math.pi
                    start_point_dis_geo = self.geod.Direct(prev_point.x - 90, prev_point.y, angle, self.start_point_dis)
                    path_point = {
                        "coord": [start_point_dis_geo["lat2"] + 90, start_point_dis_geo["lon2"]],
                        "spd": self.search_spd
                    }
                    path_point_list.append(path_point)
                if index == len(out_path) - 2:
                    spd = ves.spd
                else:
                    spd = self.search_spd
                time_cost += dist / (0.514444 * spd)
                # print(time_cost)
                prev_point = point
                path_point = {
                    "coord": [point.x, point.y],
                    "spd": spd
                }
                path_point_list.append(path_point)

            for task_number in range(len(task_points) - 1):
                task_point_number = task_points[task_number]
                tar_number = int(task_point_number)
                target_area = SusTarget(self.susTargetInfo[tar_number]["susTargetId"],
                                        self.susTargetInfo[tar_number]["susTargetArea"])
                if task_number < len(task_points) - 1:
                    next_area = SusTarget(self.susTargetInfo[int(task_points[task_number + 1])]["susTargetId"],
                                          self.susTargetInfo[int(task_points[task_number + 1])]["susTargetArea"])
                else:
                    next_area = None
                point_list = self.find_next_point(target_area, prev_point, next_area)
                for index in range(len(point_list)):
                    point = point_list[index]
                    dist = self.geod.Inverse(point.x - 90, point.y, prev_point.x - 90, prev_point.y)["s12"]
                    if index == len(point_list) - 2:
                        spd = ves.spd
                    else:
                        spd = self.search_spd
                    time_cost += dist / (0.514444 * spd)
                    prev_point = point
                    path_point = {
                        "coord": [point.x, point.y],
                        "spd": spd
                    }
                    path_point_list.append(path_point)

            path_point_dic = dict()
            path_point_dic.update({"shape": "LineString"})
            path_point_dic.update({"points": path_point_list})
            ves_output_info.update({"time": time_cost})
            self.jqtime = max(time_cost, self.jqtime)
            ves_output_info.update({"path": path_point_dic})
            output.append(deepcopy(ves_output_info))

        arguments = dict()
        arguments.update({"statusInfo": system_state})
        arguments.update({"jqtime": self.jqtime})
        arguments.update({"vesInfo": output})
        content = {"arguments": arguments}
        ves_dict.update({"content": content})

        return ves_dict

    def path_plan(self):
        start_time = time.time()
        assignment_list = list()
        if not self.work_state:
            ves_dict = self.output_json(assignment_list, self.system_state)
            return ves_dict

        planner = Planner(self.vessel_list, self.sustarget_list)
        assignment_list = planner.plan()
        # assignment_list = planner.my_plan()

        print("tasks allocation end!")
        ves_dict = self.output_json(assignment_list, self.system_state)
        end_time = time.time()
        plan_time = end_time - start_time
        print("plan time cost:", plan_time, "s")
        return ves_dict


if __name__ == "__main__":
    # path = "./input.json"
    path = "input/input_test7.json"

    with open(path, 'r', encoding="utf8") as f:
        task_ves_info = json.load(f)

    pathPlan = PathPlanner(task_ves_info)
    ves_dict_info = pathPlan.path_plan()

    json_str = json.dumps(ves_dict_info, indent=4)
    with open("output.json", 'w') as f:
        f.write(json_str)
