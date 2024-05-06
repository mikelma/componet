import numpy as np

single_tasks = [
    "hammer-v2",
    "push-wall-v2",
    "faucet-close-v2",
    "push-back-v2",
    "stick-pull-v2",
    "handle-press-side-v2",
    "push-v2",
    "shelf-place-v2",
    "window-close-v2",
    "peg-unplug-side-v2",
]

tasks = single_tasks + single_tasks


def get_task_name(task_id):
    return tasks[task_id]


def get_task(task_id, render=False):
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

    name = tasks[task_id] + "-goal-observable"

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]
    env = env_cls(seed=np.random.randint(0, 1024))

    if render:
        env.render_mode = "human"
    env._freeze_rand_vec = False

    return env


if __name__ == "__main__":
    env = get_task(0, render=True)

    for _ in range(200):
        obs, _ = env.reset()  # reset environment
        a = env.action_space.sample()  # sample an action

        # step the environment with the sampled random action
        obs, reward, terminated, truncated, info = env.step(a)

        if terminated:
            break
