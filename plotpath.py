import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import json
from model import SusTarget

COLOR_BAR = ["red", "blue", "green", "yellow"]


def plot_path(input_path, output_path):
    with open(input_path, "r", encoding='utf8') as f:
        task_dict_info = json.load(f)

    with open(output_path, 'r', encoding='utf8') as f:
        ves_dict_info = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([120.999, 121.06, 21.999, 22.03])

    status_info = ves_dict_info["content"]["arguments"]["statusInfo"]
    ves_info = ves_dict_info["content"]["arguments"]["vesInfo"]
    start_pos_info = task_dict_info["content"]["arguments"]["vesInfo"]
    target_info = task_dict_info["content"]["arguments"]["targetInfo"]
    sus_target_info = task_dict_info["content"]["arguments"]["susTargetInfo"]
    target_thread_radius = task_dict_info["content"]["arguments"]["config"]["targetThreatRadius"]
    dead_zone_width = task_dict_info["content"]["arguments"]["config"]["deadZoneWidth"]

    for i in range(len(ves_info)):
        ves_i = ves_info[i]
        ves_id = ves_i["tid"]
        ves_path = ves_i["path"]["points"]
        ves_start_pos = ves_path[0]["coord"]
        ax.text(ves_start_pos[0], ves_start_pos[1], ves_id)
        x = list()
        y = list()
        x.append(start_pos_info[i]["vesPos"][0])
        y.append(start_pos_info[i]["vesPos"][1])
        cir = Circle(xy=start_pos_info[i]["vesPos"], radius=20 / 111000)
        ax.add_patch(cir)
        for point in ves_path:
            x.append(point["coord"][0])
            y.append(point["coord"][1])
        ax.plot(x, y, color=COLOR_BAR.pop(), linestyle='dashed')
    for target_i in target_info:
        cir = Circle(xy=target_i["targetPos"], radius=target_thread_radius / 111000)
        ax.add_patch(cir)
    for sus_target_i in sus_target_info:
        sus_tar = SusTarget(sus_target_i["susTargetId"], sus_target_i["susTargetArea"], dead_zone_width)
        rec = Rectangle(xy=(sus_tar.ld_angle.x, sus_tar.ld_angle.y), width=(sus_tar.rd_angle.x - sus_tar.ld_angle.x),
                        height=(sus_tar.lu_angle.y - sus_tar.ld_angle.y))
        ax.add_patch(rec)
    plt.show()


if __name__ == "__main__":
    plot_path(input_path="input/input_test7.json", output_path="output.json")
