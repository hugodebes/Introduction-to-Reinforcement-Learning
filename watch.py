import itertools
import time

from main import (
    device,
    ENV_ID,
    LOAD_PATH,
    GAMMA,
)
from utils import make_env, load_state
from model import Network


def watch():
    """
    Function to deploy an agent and watch it play
    """
    fg_watch = True
    print("Rendering Video")
    print("device:", device)

    env = make_env(ENV_ID, fg_watch=fg_watch)
    net = Network(env, GAMMA, device).to(device)

    prev_state = load_state(LOAD_PATH)
    net.load_state_dict(prev_state["state_dict"])

    obs = env.reset()[0]
    beginning_round = True
    lives = 5
    for _ in itertools.count():
        action = net.act(obs, fg_watch=fg_watch)

        if beginning_round:
            action = 1
            beginning_round = False

        obs, _, done, _, info = env.step(action)

        if lives != info["lives"]:
            beginning_round = True
            lives = info["lives"]

        env.render()
        time.sleep(0.02)
        if done:
            obs = env.reset()
            beginning_round = True
            lives = 5


if __name__ == "__main__":
    watch()
