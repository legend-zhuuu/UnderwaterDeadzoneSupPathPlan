import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import json
from model import SusTarget
from pathplanner import PathPlanner
COLOR_BAR = ["red", "blue", "green", "yellow"]


def plot_path(input_path, output_path):
    with open(input_path, "r", encoding='utf8') as f:
        task_dict_info = json.load(f)

    pathPlanner = PathPlanner(task_dict_info)
    start_pos_info = task_dict_info["content"]["arguments"]["vesInfo"]
    target_thread_radius = task_dict_info["content"]["arguments"]["config"]["targetThreatRadius"]

    task_area = pathPlanner.task_area

    # prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([task_area.ld_angle.x - 0.001, task_area.rd_angle.x + 0.001, task_area.ld_angle.y - 0.001, task_area.lu_angle.y + 0.001])

    rec = Rectangle(xy=(task_area.ld_angle.x, task_area.ld_angle.y),
                    width=(task_area.rd_angle.x - task_area.ld_angle.x),
                    height=(task_area.lu_angle.y - task_area.ld_angle.y), color='blue', fill=False)
    ax.add_patch(rec)

    target_info = pathPlanner.target_list
    for target_i in target_info:
        cir = Circle(xy=(target_i.pos.x, target_i.pos.y), radius=target_thread_radius / 111000, color='red')
        ax.add_patch(cir)
        rec = Rectangle(xy=(target_i.ld_angle_extend.x, target_i.ld_angle_extend.y),
                        width=(target_i.rd_angle_extend.x - target_i.ld_angle_extend.x),
                        height=(target_i.lu_angle_extend.y - target_i.ld_angle_extend.y), color="red", fill=False)
        ax.add_patch(rec)
        ax.text(target_i.pos.x, target_i.pos.y, target_i.id)

    sus_target_info = pathPlanner.sustarget_list
    for sus_target_i in sus_target_info:
        x = [sus_target_i.lu_angle.x, sus_target_i.ld_angle.x, sus_target_i.rd_angle.x, sus_target_i.ru_angle.x]
        y = [sus_target_i.lu_angle.y, sus_target_i.ld_angle.y, sus_target_i.rd_angle.y, sus_target_i.ru_angle.y]
        ax.fill(x, y, color="black")
        rec = Rectangle(xy=(sus_target_i.ld_angle_extend.x, sus_target_i.ld_angle_extend.y),
                        width=(sus_target_i.rd_angle_extend.x - sus_target_i.ld_angle_extend.x),
                        height=(sus_target_i.lu_angle_extend.y - sus_target_i.ld_angle_extend.y), color="black", fill=False)
        ax.add_patch(rec)
        ax.text(sus_target_i.center.x, sus_target_i.center.y, sus_target_i.id)

    ves_info = pathPlanner.vessel_list
    for ves in ves_info:
        ves_id = ves.tid
        cir = Circle(xy=(ves.pos.x, ves.pos.y), radius=10 / 111000, color='green')
        ax.add_patch(cir)
        ax.text(ves.pos.x, ves.pos.y, ves_id)

    if output_path:
        with open(output_path, 'r', encoding='utf8') as f:
            ves_dict_info = json.load(f)
        ves_info = ves_dict_info["content"]["arguments"]["vesInfo"]

        for i in range(len(ves_info)):
            ves_i = ves_info[i]
            ves_id = ves_i["tid"]
            ves_path = ves_i["path"][0]["points"]
            x = list()
            y = list()
            x.append(start_pos_info[i]["vesPos"][0])
            y.append(start_pos_info[i]["vesPos"][1])
            for point in ves_path:
                x.append(point["coord"][0])
                y.append(point["coord"][1])
            ax.plot(x, y, color=COLOR_BAR.pop(), linestyle='dashed')
    else:
        pass

    plt.show()


if __name__ == "__main__":
    plot_path(input_path="input/input_test13.json", output_path=None)
