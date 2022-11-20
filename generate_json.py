import numpy as np
import json
import random
from plotpath import plot_path


def generate(sustar_num, target_num, ves_num=4):
    input = {}
    input.update({"id": 1})
    input.update({"method": "notice-event"})

    content = {}
    arguments = {}
    # task area
    task_area = [[121.0, 22.01852], [121.0, 22.0], [121.05556, 22.0], [121.05556, 22.01852], [121.0, 22.01852]]
    x_min = 121.00100
    x_max = 121.05456
    y_min = 22.00100
    y_max = 22.01752

    sus_tar_x_range = [40/110000, 100/110000]
    sus_tar_y_range = [10/110000, 40/110000]

    # targetInfo
    target_info = list()
    for i in range(target_num):
        target_id = 800001 + i
        target_pos = [x_min + (x_max - x_min) * random.random(), y_min + (y_max - y_min) * random.random()]
        info = {
            "targetId": target_id,
            "targetPos": target_pos,
        }
        target_info.append(info)

    # susTargetInfo
    sus_target_info = list()
    for i in range(sustar_num):
        sus_target_id = 800001 + i
        sus_target_area_pos = [x_min + (x_max - x_min) * random.random(), y_min + (y_max - y_min) * random.random()]
        dx = sus_tar_x_range[0] + (sus_tar_x_range[1] - sus_tar_x_range[0]) * random.random()
        dy = sus_tar_y_range[0] + (sus_tar_y_range[1] - sus_tar_y_range[0]) * random.random()
        sus_target_area = [[sus_target_area_pos[0], sus_target_area_pos[1]],
                           [sus_target_area_pos[0] + dx, sus_target_area_pos[1]],
                           [sus_target_area_pos[0] + dx, sus_target_area_pos[1] + dy],
                           [sus_target_area_pos[0], sus_target_area_pos[1] + dy],
                           [sus_target_area_pos[0], sus_target_area_pos[1]],
                           ]
        info = {
            "susTargetId": sus_target_id,
            "susTargetArea": sus_target_area,
        }
        sus_target_info.append(info)

    # vesInfo
    vesPosBase = [121.03000, 22.01342]
    interval = 0.002
    ves_info = list()
    for i in range(ves_num):
        tid = "03AF0" + str(101 + i)
        ves_pos = [vesPosBase[0], vesPosBase[1] - interval * i]
        sonar_width = 100
        info ={
            "tid": tid,
            "vesPos": ves_pos,
            "sonarWidth": sonar_width,
        }
        ves_info.append(info)

    # config
    config = dict()
    config.update({
        "targetThreatRadius": 70,
        "deadZoneWidth": 30,
        "startPointdis": 50,
        "speed": 5,
        "sonarLength": 20
    })

    # initial
    initial = False

    arguments.update({
        "taskArea": task_area,
        "targetInfo": target_info,
        "susTargetInfo": sus_target_info,
        "vesInfo": ves_info,
        "config": config,
        "initial": initial,
    })
    content.update({"arguments": arguments})
    input.update({"content": content})
    return input


if __name__ == "__main__":
    input_dict = generate(sustar_num=100, target_num=0, ves_num=4)
    json_str = json.dumps(input_dict, indent=4)
    with open('input/input.json', 'w') as f:
        f.write(json_str)
    plot_path(input_path="input/input.json", output_path=None)