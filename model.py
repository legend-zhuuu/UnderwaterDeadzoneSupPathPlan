import numpy as np


class Area:
    def __init__(self, area):
        self.area = area
        self.ld_angle = self.get_left_down_angle()
        self.lu_angle = self.get_left_up_angle()
        self.rd_angle = self.get_right_down_angle()
        self.ru_angle = self.get_right_up_angle()
        self.center = self.get_center()
        self.length = self.rd_angle.x - self.ld_angle.x
        self.width = self.lu_angle.y - self.ld_angle.y

    def get_center(self):
        point = self.ld_angle + self.ru_angle
        return Point([point.x / 2, point.y / 2])

    def get_left_down_angle(self):
        return Point([min(self.area[0][0], self.area[2][0]), min(self.area[0][1], self.area[1][1])])

    def get_left_up_angle(self):
        return Point([min(self.area[0][0], self.area[2][0]), max(self.area[0][1], self.area[1][1])])

    def get_right_down_angle(self):
        return Point([max(self.area[0][0], self.area[2][0]), min(self.area[0][1], self.area[1][1])])

    def get_right_up_angle(self):
        return Point([max(self.area[0][0], self.area[2][0]), max(self.area[0][1], self.area[1][1])])


class SusTarget(Area):
    def __init__(self, sus_target_id: str, sus_target_area):
        super().__init__(sus_target_area)
        self.id = sus_target_id
        self.type = self.judge_type()

    def judge_type(self):
        if self.length > self.width:
            return "west_east"
        else:
            return "south_north"


class Point:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def __add__(self, other):
        return Point([self.x + other.x, self.y + other.y])

    def __sub__(self, other):
        return Point([self.x - other.x, self.y - other.y])


class Vessel:
    def __init__(self, tid: str, ves_pos: list, sonar_width: float, spd: float):
        self.tid = tid
        self.pos = Point(np.array(ves_pos))
        self.sonarWidth = sonar_width
        self.spd = spd


class Target:
    def __init__(self, target_id: str, target_pos: list):
        self.id = target_id
        self.pos = Point(np.array(target_pos))
