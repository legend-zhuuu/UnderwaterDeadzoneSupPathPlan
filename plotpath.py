import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import json
from model import SusTarget

COLOR_BAR = ["red", "blue", "green", "yellow"]


def plot_path(input_path, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([120.999, 121.06, 21.999, 22.03])

    with open(input_path, "r", encoding='utf8') as f:
        task_dict_info = json.load(f)

    start_pos_info = task_dict_info["content"]["arguments"]["vesInfo"]
    target_info = task_dict_info["content"]["arguments"]["targetInfo"]
    sus_target_info = task_dict_info["content"]["arguments"]["susTargetInfo"]
    target_thread_radius = task_dict_info["content"]["arguments"]["config"]["targetThreatRadius"]
    dead_zone_width = task_dict_info["content"]["arguments"]["config"]["deadZoneWidth"]
    ves_info = task_dict_info["content"]["arguments"]["vesInfo"]

    for target_i in target_info:
        cir = Circle(xy=target_i["targetPos"], radius=target_thread_radius / 111000, color='red')
        ax.add_patch(cir)
        ax.text(target_i["targetPos"][0], target_i["targetPos"][1], target_i["targetId"])
    for sus_target_i in sus_target_info:
        sus_tar = SusTarget(sus_target_i["susTargetId"], sus_target_i["susTargetArea"], dead_zone_width)
        x = [x[0] for x in sus_target_i["susTargetArea"]]
        y = [x[1] for x in sus_target_i["susTargetArea"]]
        ax.fill(x, y, color="black")
        rec = Rectangle(xy=(sus_tar.ld_angle.x, sus_tar.ld_angle.y), width=(sus_tar.rd_angle.x - sus_tar.ld_angle.x),
                        height=(sus_tar.lu_angle.y - sus_tar.ld_angle.y), color="black")
        # ax.add_patch(rec)
        ax.text(sus_tar.center.x, sus_tar.center.y, sus_tar.id)
    for ves in ves_info:
        ves_id = ves["tid"]
        cir = Circle(xy=ves["vesPos"], radius=10 / 111000, color='green')
        ax.add_patch(cir)
        ax.text(ves["vesPos"][0], ves["vesPos"][1], ves_id)

    if output_path:
        with open(output_path, 'r', encoding='utf8') as f:
            ves_dict_info = json.load(f)
        ves_info = ves_dict_info["content"]["arguments"]["vesInfo"]

        for i in range(len(ves_info)):
            ves_i = ves_info[i]
            ves_id = ves_i["tid"]
            ves_path = ves_i["path"]["points"]
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
    plot_path(input_path="input/input_test14.json", output_path=None)
