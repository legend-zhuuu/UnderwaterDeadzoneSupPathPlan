from pathplanner import PathPlanner
import json
from plotpath import plot_path


if __name__ == "__main__":
    input_path = "input/input.json"
    output_path = "output.json"

    with open(input_path, 'r', encoding="utf8") as f:
        task_ves_info = json.load(f)

    pathPlan = PathPlanner(task_ves_info)
    ves_dict_info = pathPlan.path_plan()

    json_str = json.dumps(ves_dict_info, indent=4)
    with open(output_path, 'w') as f:
        f.write(json_str)

    if ves_dict_info["content"]["arguments"]["statusInfo"]["inputState"] != 1:
        plot_path(input_path, output_path)
