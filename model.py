import numpy as np
from geographiclib.geodesic import Geodesic


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
    def __init__(self, sus_target_id: str, sus_target_area, dead_zone_width):
        super().__init__(sus_target_area)
        self.id = sus_target_id
        self.type = self.judge_type()
        self.dead_zone_width = dead_zone_width
        self.plus = 10
        self.ld_angle_extend = None
        self.lu_angle_extend = None
        self.rd_angle_extend = None
        self.ru_angle_extend = None
        self.geod = Geodesic.WGS84
        self.geod = Geodesic(6378388, 1 / 297.0)
        self.get_angle_extend()

    def judge_type(self):
        if self.length > self.width:
            return "west_east"
        else:
            return "south_north"

    def get_angle_extend(self):
        x, y = self.ld_angle.x, self.ld_angle.y
        length_geo = self.geod.Direct(x - 90, y, 90, self.dead_zone_width / 2 + self.plus)
        length = length_geo["lon2"] - y
        width_geo = self.geod.Direct(x - 90, y, 0, self.dead_zone_width / 2 + self.plus)
        width = width_geo["lat2"] + 90 - x
        self.ld_angle_extend = self.ld_angle - Point([width, length])
        self.lu_angle_extend = self.lu_angle - Point([width, -length])
        self.rd_angle_extend = self.rd_angle - Point([-width, length])
        self.ru_angle_extend = self.ru_angle - Point([-width, -length])


class Target:
    def __init__(self, target_id: str, target_pos: list, threat_radius, threat_radius_plus):
        self.id = target_id
        self.pos = Point(np.array(target_pos))
        self.target_threat_radius = threat_radius
        self.target_threat_radius_plus = threat_radius_plus
        self.ld_angle_extend = None
        self.lu_angle_extend = None
        self.rd_angle_extend = None
        self.ru_angle_extend = None
        self.rd_angle = None
        self.ru_angle = None
        self.ld_angle = None
        self.lu_angle = None
        self.geod = Geodesic.WGS84
        self.geod = Geodesic(6378388, 1 / 297.0)
        self.get_angle_extend()

    def get_angle_extend(self):
        geo1 = self.geod.Direct(self.pos.x - 90, self.pos.y, -45,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo2 = self.geod.Direct(self.pos.x - 90, self.pos.y, 45,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo3 = self.geod.Direct(self.pos.x - 90, self.pos.y, -135,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        geo4 = self.geod.Direct(self.pos.x - 90, self.pos.y, 135,
                                self.target_threat_radius * 1.414 + self.target_threat_radius_plus)
        self.rd_angle_extend = Point([geo1["lat2"] + 90, geo1["lon2"]])
        self.ru_angle_extend = Point([geo2["lat2"] + 90, geo2["lon2"]])
        self.ld_angle_extend = Point([geo3["lat2"] + 90, geo3["lon2"]])
        self.lu_angle_extend = Point([geo4["lat2"] + 90, geo4["lon2"]])
        self.rd_angle = self.rd_angle_extend
        self.ru_angle = self.ru_angle_extend
        self.ld_angle = self.ld_angle_extend
        self.lu_angle = self.lu_angle_extend
