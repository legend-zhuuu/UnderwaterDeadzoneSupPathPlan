"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np
import matplotlib.pyplot as plt
from model import Point
from utils import compute_dist
from collections import deque


class APFPlanner:
    def __init__(self, obs_list, sus_list, spd, geod):
        self.obs_list = obs_list
        self.sus_list = sus_list
        self.KP = 5.0
        self.ETA = 1.
        self.resp = 10 / 100000
        self.threshold = 500 / 100000
        self.spd = spd
        self.geod = geod
        self.current_pos = None

    def calc_potential_force(self, point, goal):
        attr_force = self.calc_attractive_force(point, goal)
        repu_force = self.calc_repulsive_force(point)
        return attr_force + repu_force

    def calc_attractive_force(self, point, goal):
        """
        计算引力
        """
        direction_vector = goal - point
        direction_vector = np.array([direction_vector.x, direction_vector.y])
        dist = compute_dist(goal, point)
        if dist > self.threshold:
            return Point(self.KP * direction_vector)
        else:
            return Point(self.KP * direction_vector * self.threshold / dist)

    def calc_repulsive_force(self, point):
        rep_force = Point([0., 0.])
        for sus in self.sus_list:
            dist = compute_dist(point, sus.center)
            direction_vector = sus.center - point
            direction_vector = np.array([direction_vector.x, direction_vector.y])
            sus_threshold = compute_dist(sus.lu_angle, sus.center)
            if dist >= sus_threshold:
                continue
            else:
                rep_force -= Point(self.ETA * (1/dist - 1/sus_threshold) * (1/dist)**2 * direction_vector / dist)

        for obs in self.obs_list:
            dist = compute_dist(point, obs.pos)
            direction_vector = obs.pos - point
            direction_vector = np.array([direction_vector.x, direction_vector.y])
            obs_threshold = compute_dist(obs.lu_angle_extend, obs.pos)
            if dist >= obs_threshold:
                continue
            else:
                rep_force -= Point(self.ETA * (1 / dist - 1 / obs_threshold) * (1 / dist) ** 2 * direction_vector / dist)
        return rep_force

    def get_APF_path(self, start, end, direct):
        d = compute_dist(start, end)
        self.current_pos = start
        pos_history = list()
        while d > self.resp:
            force = self.calc_potential_force(self.current_pos, end)
            _direction = np.arctan2(force.y, force.x)
            direct += np.clip((_direction - direct), -np.pi / 6, np.pi / 6) * 1.0
            self.current_pos = Point([
                self.geod.Direct(self.current_pos.y, self.current_pos.x, 90 - direct * 180 / np.pi, self.spd * 1.0)["lon2"],
                self.geod.Direct(self.current_pos.y, self.current_pos.x, 90 - direct * 180 / np.pi, self.spd * 1.0)["lat2"]
            ])
            # self.current_pos.print()
            pos_history.append(self.current_pos)
            d = compute_dist(self.current_pos, end)
            if self.stand_still(pos_history):
                print("Oscillation detected", self.current_pos.print())
                return False, pos_history
        pos_history.append(end)
        return True, pos_history

    def stand_still(self, pos_history):
        x_sum = 0
        y_sum = 0
        for pos in pos_history[-10:]:
            x_sum += pos.x
            y_sum += pos.y
        x_err = abs((x_sum / 10 - self.current_pos.x))
        y_err = abs((y_sum / 10 - self.current_pos.y))
        if (x_err < 10 / 100000) and (y_err < 10 / 100000):
            return True
        else:
            return False
