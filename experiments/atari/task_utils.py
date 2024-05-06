TASKS = {
    "ALE/SpaceInvaders-v5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ALE/Freeway-v5": [0, 1, 2, 3, 4, 5, 6, 7],
}


def parse_name_info(name):
    fields = name.split("__")
    if "SpaceInvaders" in fields[0]:
        env = "ALE/SpaceInvaders-v5"
    elif "Freeway" in fields[0]:
        env = "ALE/Freeway-v5"
    mode = int(fields[0].split("_")[-1])
    algorithm = fields[1]
    seed = int(fields[3])
    return env, mode, algorithm, seed


def path_from_other_mode(base_path, new_mode):
    sep_idx = base_path.index("_")
    double_sep_idx = base_path.index("__")
    new_path = base_path[: sep_idx + 1] + str(new_mode) + base_path[double_sep_idx:]
    return new_path
