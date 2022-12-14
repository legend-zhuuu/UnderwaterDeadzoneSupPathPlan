import numpy as np
import math
import argparse
import scipy.io as sio
from datetime import datetime
import os
import sys
import time
import json
from copy import deepcopy
from itertools import permutations
from geographiclib.geodesic import Geodesic
from matplotlib import pyplot as plt

FILE = os.path.dirname(__file__)
if FILE not in sys.path:
    sys.path.append(FILE)

from simple_planner import Planner
from model import Point, Vessel, SusTarget, Target, Area
from utils import compute_dist
from algorithms import PSO


class PathPlanner:
    def __init__(self, task_ves_info):
        # 导入json文件中的任务和舰艇信息
        self.task_ves_info = task_ves_info

        # system arguments
        self.system_state = {
            "workState": True,
            "inputState": 0,
            "outputState": True,
            "msg": "ok"
        }

        # geographiclib
        self.geod = Geodesic.WGS84
        self.geod = Geodesic(6378388, 1 / 297.0)

        self.sonar_length_plus = 5
        self.dead_zone_width_plus = 5
        self.target_threat_radius_plus = 10

        # information
        self.target_number = len(self.task_ves_info["content"]["arguments"]["susTargetInfo"])
        self.agent_number = len(self.task_ves_info["content"]["arguments"]["vesInfo"])

        try:
            self.load_target_threat_radius()
            self.load_dead_zone_width()
            self.load_start_point_dis()
            self.load_search_spd()
            self.load_sonar_length()
        except KeyError:
            self.system_state["msg"] = "{} information are incomplete.".format(self.system_state["msg"])
            self.system_state["workState"] = False
            self.system_state["inputState"] = 1
            self.system_state["outputState"] = False

        self.turn_radius = 40
        self.degree = 135
        self.initial = self.task_ves_info["content"]["arguments"]["initial"]

        self.jqtime = 0

        self.target_list = list()
        self.sustarget_list = list()
        self.vessel_list = list()

        # load information
        if self.system_state["workState"]:
            try:
                self.load_global_info()
                self.load_targets_pos_info()
                self.load_sus_target_info()
                self.load_ves_info()
                self.check_input_valid()
            except KeyError:
                self.system_state["msg"] = "{} information are incomplete.".format(self.system_state["msg"])
                self.system_state["workState"] = False
                self.system_state["inputState"] = 1
                self.system_state["outputState"] = False
        # output
        self.ves_dict = dict()

    def load_target_threat_radius(self):
        self.system_state["msg"] = "targetThreatRadius"
        self.target_threat_radius = self.task_ves_info["content"]["arguments"]["config"]["targetThreatRadius"]

    def load_dead_zone_width(self):
        self.system_state["msg"] = "deadZoneWidth"
        self.dead_zone_width = self.task_ves_info["content"]["arguments"]["config"]["deadZoneWidth"]
        self.dead_zone_width += self.dead_zone_width_plus

    def load_start_point_dis(self):
        self.system_state["msg"] = "startPointdis"
        self.start_point_dis = self.task_ves_info["content"]["arguments"]["config"]["startPointdis"]

    def load_search_spd(self):
        self.system_state["msg"] = "speed"
        self.search_spd = self.task_ves_info["content"]["arguments"]["config"]["speed"]

    def load_sonar_length(self):
        self.system_state["msg"] = "sonarLength"
        self.sonar_length = self.task_ves_info["content"]["arguments"]["config"]["sonarLength"]

    def load_global_info(self):
        self.system_state["msg"] = "taskArea"
        self.task_area = Area(self.task_ves_info["content"]["arguments"]["taskArea"])

    def load_targets_pos_info(self):
        self.system_state["msg"] = "targetInfo"
        self.targetInfo = self.task_ves_info["content"]["arguments"]["targetInfo"]
        for tar_info in self.targetInfo:
            target = Target(tar_info["targetId"], tar_info["targetPos"], self.target_threat_radius,
                            self.target_threat_radius_plus)
            self.target_list.append(target)

    def load_sus_target_info(self):
        self.system_state["msg"] = "susTargetInfo"
        self.susTargetInfo = self.task_ves_info["content"]["arguments"]["susTargetInfo"]
        for sus_tar_info in self.susTargetInfo:
            sus_target = SusTarget(sus_tar_info["susTargetId"], sus_tar_info["susTargetArea"], self.dead_zone_width)
            self.sustarget_list.append(sus_target)

    def load_ves_info(self):
        self.system_state["msg"] = "vesInfo"
        self.vesInfo = self.task_ves_info["content"]["arguments"]["vesInfo"]
        for ves_info in self.vesInfo:
            vessel = Vessel(ves_info["tid"], ves_info["vesPos"], ves_info["sonarWidth"])
            self.vessel_list.append(vessel)

    def check_input_valid(self):
        valid_targetid_input = [800001, 899999]
        valid_sustargetid_input = [800001, 899999]
        sonar_range = [0, 450]

        config_range = {
            "targetThreatRadius": [0, 5000],
            "deadZoneWidth": [0, 50],
            "startPointdis": [0, 1000],
            "speed": [0, 40],
            "sonarLength": [0, 100],
        }

        # target
        for tar in self.target_list:
            if not valid_targetid_input[0] <= tar.id <= valid_targetid_input[1]:
                self.system_state["msg"] = "targetId input is out of range"
                self.system_state["inputState"] = 2

        # sustarget
        if len(self.sustarget_list) == 0:
            self.system_state["msg"] = "The quantity of sus target area is zero"
            self.system_state["inputState"] = 1
            self.system_state["outputState"] = False
            self.system_state["workState"] = False
            return
        for sustar in self.sustarget_list:
            x, y = sustar.ld_angle.x, sustar.ld_angle.y
            sonar_length_geo = self.geod.Direct(y, x, 0, self.sonar_length)
            sonar_length = sonar_length_geo["lat2"] - y
            if not self.point_in_area(sustar.ld_angle, self.task_area) or not self.point_in_area(sustar.ru_angle, self.task_area) or \
                    (sustar.rd_angle.x + sonar_length > self.task_area.rd_angle.x):
                self.system_state["msg"] = "{} sus target area is invalid".format(sustar.id)
                self.system_state["inputState"] = 2
                self.system_state["outputState"] = False
                self.system_state["workState"] = False
                return
            if not valid_sustargetid_input[0] <= sustar.id <= valid_sustargetid_input[1]:
                self.system_state["msg"] = "susTargetId input is out of range"
                self.system_state["inputState"] = 2

        # ves
        if len(self.vessel_list) == 0:
            self.system_state["msg"] = "The quantity of vessel is zero"
            self.system_state["inputState"] = 1
            self.system_state["outputState"] = False
            self.system_state["workState"] = False
            return
        for ves in self.vessel_list:
            if not sonar_range[0] <= ves.sonarWidth <= sonar_range[1]:
                self.system_state["msg"] = "sonar input is out of range"
                self.system_state["inputState"] = 2
                self.system_state["outputState"] = False
                self.system_state["workState"] = False
                return

        # config input
        for config_key, config_value in self.task_ves_info["content"]["arguments"]["config"].items():
            if not config_range[config_key][0] <= config_value <= config_range[config_key][1]:
                self.system_state["msg"] = "{} input is out of range".format(config_key)
                self.system_state["inputState"] = 2
                self.system_state["outputState"] = False
                self.system_state["workState"] = False
                return

        # input valid
        self.system_state["msg"] = "ok"

    def print_item(self, item):
        print(getattr(item, "__class__"), item.ld_angle_extend.x, item.ld_angle_extend.y)

    def print_points_list(self, points_list):
        i = 0
        for point in points_list:
            print(i, point.x, point.y)
            i += 1

    def empty_ves_dict(self):
        self.ves_dict = dict()
        self.ves_dict.update({"id": 1})
        self.ves_dict.update({"method": "notice-event"})
        arguments = dict()
        arguments.update({"statusInfo": self.system_state})
        arguments.update({"jqtime": 0})
        arguments.update({"vesInfo": []})
        content = {"arguments": arguments}
        self.ves_dict.update({"content": content})

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
        if getattr(getattr(area, '__class__'), "__name__") == "Area":
            if area.ld_angle.x < point.x < area.rd_angle.x and area.ld_angle.y < point.y < area.lu_angle.y:
                return True
        else:
            if area.ld_angle_extend.x < point.x < area.rd_angle_extend.x and area.ld_angle_extend.y < point.y < area.lu_angle_extend.y:
                return True
        return False

    def pass_through_sus_tar_or_obs(self, start_point, end_point):
        for item in self.sustarget_list + self.target_list:
            if self.point_in_area(start_point, item) or self.point_in_area(end_point, item):
                return item, True
            if self.is_intersec(item.ld_angle_extend, item.ru_angle_extend, start_point, end_point) or \
                    self.is_intersec(item.lu_angle_extend, item.rd_angle_extend, start_point, end_point):
                return item, True
        return [], False

    def path_success(self, point_list, prev_point):
        for index in range(len(point_list) - 1):
            if index == 0:
                p0 = prev_point
            else:
                p0 = p1
            p1 = point_list[index]
            p2 = point_list[index + 1]
            item, flag = self.pass_through_sus_tar_or_obs(p1, p2)
            if flag:
                return p0, p1, p2, item, True
        return None, None, None, None, False

    def out_map(self, point1, point2):
        if point1.x < self.task_area.ld_angle.x or point1.x > self.task_area.ru_angle.x or \
                point1.y < self.task_area.ld_angle.y or point1.y > self.task_area.ru_angle.y or \
                point2.x < self.task_area.ld_angle.x or point2.x > self.task_area.ru_angle.x or \
                point2.y < self.task_area.ld_angle.y or point2.y > self.task_area.ru_angle.y:
            return True
        return False

    def candi_is_legal(self, candi_pair):
        p1, p2 = candi_pair
        _, obs_flag = self.pass_through_sus_tar_or_obs(p1, p2)
        out_map_flag = self.out_map(p1, p2)
        if not obs_flag and not out_map_flag:
            return True
        return False

    def charge_item_neighbor(self, item):
        item_list = list()
        pos = item.pos
        for tar in self.target_list:
            dist = np.linalg.norm([pos.x - tar.pos.x, pos.y - tar.pos.y])
            e = 2 * (self.target_threat_radius + self.target_threat_radius_plus) / 111000
            if dist < e:
                item_list.append(tar)
        return item_list

    def merge_item_list(self, item_list):
        if len(item_list) == 1:
            return item_list[0]
        x_min = item_list[0].ld_angle_extend.x
        x_max = item_list[0].ru_angle_extend.x
        y_min = item_list[0].ld_angle_extend.y
        y_max = item_list[0].ru_angle_extend.y
        for item in item_list:
            if item.ld_angle_extend.x < x_min:
                x_min = item.ld_angle_extend.x
            if item.ld_angle_extend.y < y_min:
                y_min = item.ld_angle_extend.y
            if item.ru_angle_extend.x > x_max:
                x_max = item.ru_angle_extend.x
            if item.ru_angle_extend.y > y_max:
                y_max = item.ru_angle_extend.y
        name = ' '
        new_sus_area = SusTarget(name.join([str(item.id) for item in item_list]), [
            [x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], 0)
        return new_sus_area

    def insert_path_point(self, prev_point, start_point, end_point, item):
        insert_obs_list = list()
        angle_point_list = [item.ld_angle_extend, item.lu_angle_extend, item.rd_angle_extend, item.ru_angle_extend]

        for angle_point in angle_point_list:
            if (not self.is_intersec(start_point, angle_point, angle_point_list[0], angle_point_list[3])) and \
                    (not self.is_intersec(start_point, angle_point, angle_point_list[1], angle_point_list[2])) and \
                    (not self.is_intersec(angle_point, end_point, angle_point_list[0], angle_point_list[3])) and \
                    (not self.is_intersec(angle_point, end_point, angle_point_list[1], angle_point_list[2])) and \
                    self.is_obtuse(start_point, prev_point, angle_point) and \
                    self.is_obtuse(angle_point, start_point, end_point):
                return [angle_point]
        else:
            _angle_point_list = angle_point_list.copy()
            _prev_point = prev_point
            out_flag = False
            while not out_flag:
                for index in range(len(_angle_point_list)):
                    point = _angle_point_list[index]
                    if self.is_obtuse(start_point, _prev_point, point) and (
                            not self.is_intersec(start_point, point, angle_point_list[0], angle_point_list[3])) and \
                            (not self.is_intersec(start_point, point, angle_point_list[1],
                                                  angle_point_list[2])):
                        insert_obs_list.append(point)
                        _prev_point = start_point
                        start_point = point
                        _angle_point_list.pop(index)
                        break
                if self.is_obtuse(start_point, _prev_point, end_point) and (
                        not self.is_intersec(start_point, end_point, angle_point_list[0], angle_point_list[3])) and \
                        (not self.is_intersec(start_point, end_point, angle_point_list[1],
                                              angle_point_list[2])):
                    out_flag = True
                # time.sleep(10)
        return insert_obs_list

    def insert_acute_path_point(self, point1, point2, point3):
        insert_point_list = list()

        _lambda = (point1.x - point2.x) * (point3.x - point2.x) + (point1.y - point2.y) * (point3.y - point2.y) / (
                (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
        H = point2 + Point([_lambda * (point1 - point2).x, _lambda * (point1 - point2).y])
        vec1 = point3 - H  # HP3
        # vec_len = np.linalg.norm([vec1.x, vec1.y])
        # norm_vec = Point([vec1.x / vec_len, vec1.y / vec_len])
        angle = math.atan2(vec1.y, vec1.x) * 180 / math.pi
        vec2 = point2 - point1  # P1P2
        sign = np.sign(vec2.x * vec1.y - vec2.y * vec1.x)  # P1P2 cross HP3
        direction = math.atan2(vec2.y, vec2.x) * 180 / math.pi
        circle_sample = list()
        prev_point = point2
        step = 30
        length = np.sqrt(2 * self.turn_radius ** 2 - 2 * (self.turn_radius ** 2) * np.cos(step * np.pi / 180))
        for deg in range(step, 180, step):
            _direction = 90 - direction - sign * deg
            circle_sample.append(Point([
                self.geod.Direct(prev_point.y, prev_point.x, _direction, length)["lon2"],
                self.geod.Direct(prev_point.y, prev_point.x, _direction, length)["lat2"]
            ]))
            prev_point = circle_sample[-1]
        #
        # x, y = [p.x for p in circle_sample], [p.y for p in circle_sample]
        # x = [point1.x, point2.x] + x
        # y = [point1.y, point2.y] + y
        # plt.plot(x, y)
        # plt.show()
        prev_point = point1
        start_point = point2
        end_point = point3
        flag = False
        while not flag:
            try:
                sample_point = circle_sample.pop(0)
            except IndexError:
                print("Error! run out of sample!")
                flag = True
            for tar in self.target_list:
                if self.point_in_area(sample_point, tar):  # todo: in sus area?
                    print("path in target area, try to avoid it.")
                    flag = True
                    break
            if not flag:
                if not self.is_large_than_degree(sample_point, start_point, end_point):
                    insert_point_list.append(sample_point)
                    start_point = sample_point
                else:
                    insert_point_list.append(sample_point)
                    return insert_point_list
            else:
                return self.insert_path_point(point1, point2, point3, tar)

    @staticmethod
    def is_obtuse(A, B, C):
        v1 = np.array([B.x - A.x, B.y - A.y])
        v2 = np.array([C.x - A.x, C.y - A.y])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos < 1e-3:
            return True
        return False

    def is_large_than_degree(self, A, B, C):
        v1 = np.array([B.x - A.x, B.y - A.y])
        v2 = np.array([C.x - A.x, C.y - A.y])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos < np.cos(self.degree * np.pi / 180):
            return True
        return False

    def acute2obtuse(self, prev_point, out_point_list):
        path_list = [prev_point] + out_point_list  # 4
        _path_list = path_list
        for index in range(len(path_list) - 2):
            point1 = path_list[index]
            point2 = path_list[index + 1]
            point3 = path_list[index + 2]
            if not self.is_obtuse(point2, point1, point3):
                if index == 0:
                    insert_point = self.insert_acute_path_point(point1, point2, point3)
                    ip = _path_list.index(point2)
                    _path_list = _path_list[:ip + 1] + insert_point + _path_list[ip + 1:]
                elif index == len(path_list) - 3:  # index = 1
                    insert_point = self.insert_acute_path_point(point3, point2, point1)
                    insert_point = insert_point[::-1]
                    ip = _path_list.index(_path_list[-3])
                    _path_list = _path_list[:ip + 1] + insert_point + _path_list[ip + 1:]
                else:
                    print("acute appear in middle path!!")
        return _path_list[1:]

    def find_next_point(self, prev_point, start_point, next_area):
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

            length_geo = self.geod.Direct(y, x, 0, self.dead_zone_width / 2)
            sonar_length_geo = self.geod.Direct(y, x, 90, self.sonar_length + self.sonar_length_plus)
            length = length_geo["lat2"] - y
            sonar_length = sonar_length_geo["lon2"] - x
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
            width_geo = self.geod.Direct(y, x, 90, self.dead_zone_width / 2)
            sonar_length_geo = self.geod.Direct(y, x, 0, self.sonar_length + self.sonar_length_plus)
            width = width_geo["lon2"] - x
            sonar_length = sonar_length_geo["lat2"] - y
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
            self.system_state["workState"] = False
            self.system_state["outputState"] = False
            self.empty_ves_dict()
            self.system_state["msg"] = "Path plan error! {} doesn't have a feasible path!".format(next_area.id)
            return None
        else:
            perform_list = list()
            out_flag = False
            for candi in candi_point_list:
                # candi = point1 + point2
                item, item_flag = self.pass_through_sus_tar_or_obs(start_point, candi[0])
                dist = np.linalg.norm([start_point.x - candi[0].x, start_point.y - candi[0].y])
                perform_list.append([candi, item_flag, dist, item])

            perform_list.sort(key=lambda x: x[2])
            candi, item = perform_list[0][0], perform_list[0][-1]
            out_point_list = [start_point] + candi
            out_point_list = self.acute2obtuse(prev_point, out_point_list)
            _prev_point = prev_point
            # out_flag = True
            while not out_flag:
                _prev_point, p1, p2, item, item_flag = self.path_success(out_point_list, _prev_point)
                if not item_flag:
                    out_flag = True
                else:
                    index = out_point_list.index(p1)
                    if getattr(getattr(item, '__class__'), "__name__") == "Target":
                        item_list = self.charge_item_neighbor(item)
                        item = self.merge_item_list(item_list)
                    insert_obs_point_list = self.insert_path_point(_prev_point, p1, p2, item)
                    out_point_list = out_point_list[0:index + 1] + insert_obs_point_list + out_point_list[index + 1:]
        return out_point_list[1:]

    def find_nearest_point_index(self, point, point_list):
        for i in range(len(point_list)):
            if compute_dist(Point(point), Point(point_list[i])) < 1e-3:
                return i
        return 0

    def sorted_by_x(self, assignment_list):
        sorted_assignment_list = list()
        for assignment in assignment_list:
            # from left to right
            x_list = [self.sustarget_list[int(_x)].center.x for _x in assignment]
            x_list, assignment = zip(*sorted(zip(x_list, assignment)))
            # sorted_assignment_list.append(assignment)
            n = len(assignment)
            _assignment = list()
            dist_matrix = np.ones([n, n])
            for i in range(n-1):
                for j in range(i+1, n):
                    point1 = self.sustarget_list[assignment[i]].center
                    point2 = self.sustarget_list[assignment[j]].center
                    dist_matrix[i][j] = compute_dist(point1, point2)
                    dist_matrix[j][i] = compute_dist(point1, point2)
            current_id = 0
            current_area = assignment[current_id]
            _assignment.append(current_area)
            while len(_assignment) < n:
                dist_min = min(dist_matrix[current_id])
                next_id = np.where(dist_matrix[current_id] == dist_min)[0]
                dist_matrix[current_id] = np.ones_like(dist_matrix[current_id])
                dist_matrix[:, current_id] = np.ones_like(dist_matrix[:, current_id])
                current_id = int(next_id)
                _assignment.append(assignment[current_id])
            sorted_assignment_list.append(_assignment)
        return sorted_assignment_list

    def sorted_by_PSO(self, assignment_list):
        sorted_assignment_list = list()
        for assignment in assignment_list:
            ass_data_list = [list(self.sustarget_list[j].center.to_numpy()) for j in assignment]
            ass_data = np.array(ass_data_list)
            model = PSO(num_city=ass_data.shape[0], data=ass_data.copy())
            Best_path, Best = model.run()
            new_list = list()
            for point in Best_path:
                idx_ass = assignment[self.find_nearest_point_index(point, ass_data_list)]
                new_list.append(idx_ass)
            sorted_assignment_list.append(new_list)
        return sorted_assignment_list

    def remake_assignment_list(self, assignment_list):
        # sorted_assignment_list = self.sorted_by_x(assignment_list)
        sorted_assignment_list = self.sorted_by_PSO(assignment_list)
        # 四个船路径不交错
        # 枚举
        new_assignment_list = list()
        # four vessel position
        ves_start_point = [ves.pos for ves in self.vessel_list]

        for p in permutations(sorted_assignment_list):
            _ves_start_point = ves_start_point.copy()[:len(assignment_list)]
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
        print(assignment_list)

        # output
        self.ves_dict.update({"id": 1})
        self.ves_dict.update({"method": "notice-event"})
        output = list()
        for i in range(len(assignment_list)):
            if not self.system_state["workState"]:
                break
            time_cost = 0
            sus_target_id_list = list()
            ves = self.vessel_list[i]
            ves_id = ves.tid
            ves_output_info = dict()
            ves_output_info.update({"tid": ves_id})

            path_point_list = list()
            ves_start_pos = ves.pos

            task_points = assignment_list[i]
            start_point = ves_start_pos

            first_area_info = self.susTargetInfo[int(task_points[0])]
            first_area = SusTarget(first_area_info["susTargetId"], first_area_info["susTargetArea"],
                                   self.dead_zone_width)
            sus_target_id_list.append(first_area.id)

            vec = first_area.pos - start_point
            angle = math.atan2(vec.y, vec.x) * 180 / math.pi
            start_point_dis_geo = self.geod.Direct(start_point.y, start_point.x, 90 - angle,
                                                   self.start_point_dis)
            start_point_dis = Point([start_point_dis_geo["lon2"], start_point_dis_geo["lat2"]])
            path_point = {
                "coord": [start_point_dis.x, start_point_dis.y],
                "spd": self.search_spd
            }
            path_point_list.append(path_point)

            out_path = self.find_next_point(start_point, start_point_dis, first_area)
            if not out_path:
                self.system_state["msg"] = "{} can't find a feasible path".format(ves_id)
                self.system_state["outputState"] = False
                self.system_state["workState"] = False
                break

            for index in range(len(out_path)):
                point = out_path[index]
                dist = self.geod.Inverse(point.y, point.x, start_point.y, start_point.x)["s12"]
                time_cost += dist / (0.514444 * self.search_spd)
                start_point = point
                path_point = {
                    "coord": [point.x, point.y],
                    "spd": self.search_spd
                }
                path_point_list.append(path_point)
            prev_point = out_path[-2]

            for task_number in range(len(task_points) - 1):
                task_point_number = task_points[task_number]
                tar_number = int(task_point_number)
                target_area = SusTarget(self.susTargetInfo[tar_number]["susTargetId"],
                                        self.susTargetInfo[tar_number]["susTargetArea"], self.dead_zone_width)
                if task_number < len(task_points) - 1:
                    next_area = SusTarget(self.susTargetInfo[int(task_points[task_number + 1])]["susTargetId"],
                                          self.susTargetInfo[int(task_points[task_number + 1])]["susTargetArea"],
                                          self.dead_zone_width)
                    sus_target_id_list.append(next_area.id)
                else:
                    next_area = None
                out_path = self.find_next_point(prev_point, start_point, next_area)
                if not out_path:
                    self.system_state["msg"] = "{} can't find a feasible path".format(ves_id)
                    self.system_state["outputState"] = False
                    self.system_state["workState"] = False
                    break

                for index in range(len(out_path)):
                    point = out_path[index]
                    dist = self.geod.Inverse(point.y, point.x, start_point.y, start_point.x)["s12"]
                    time_cost += dist / (0.514444 * self.search_spd)
                    start_point = point
                    path_point = {
                        "coord": [point.x, point.y],
                        "spd": self.search_spd
                    }
                    path_point_list.append(path_point)
                prev_point = out_path[-2]

            path_point_dic = dict()
            path_point_dic.update({"shape": "LineString"})
            path_point_dic.update({"points": path_point_list})
            ves_output_info.update({"time": time_cost})
            self.jqtime = max(time_cost, self.jqtime)
            ves_output_info.update({"path": [path_point_dic]})
            output.append(deepcopy(ves_output_info))
            print("{} complete path plan. Search sustarget id:{}".format(ves_id, sus_target_id_list))

        arguments = dict()
        arguments.update({"statusInfo": system_state})
        arguments.update({"jqTime": self.jqtime})
        arguments.update({"vesInfo": output})
        content = {"arguments": arguments}
        self.ves_dict.update({"content": content})

        return self.ves_dict

    def path_plan(self):
        start_time = time.time()
        assignment_list = list()
        if not self.system_state["workState"]:
            ves_dict = self.output_json(assignment_list, self.system_state)
            return ves_dict

        planner = Planner(self.vessel_list, self.sustarget_list)
        assignment_list = planner.plan("c_fuzzy")
        # assignment_list = planner.my_plan()

        print("tasks allocation end!")
        ves_dict = self.output_json(assignment_list, self.system_state)
        end_time = time.time()
        plan_time = end_time - start_time
        print("plan time cost:", plan_time, "s")
        return ves_dict
