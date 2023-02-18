import numpy as np
import random
import torch

from collections import deque
from torch.utils.tensorboard import SummaryWriter
from model import Network
from utils import make_env, make_replay_buffer, save_state, load_state
from per import PrioritizedReplayBuffer
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID = "ALE/Breakout-v5"
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = int(1e6)
MIN_REPLAY_SIZE = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5e4
TARGET_UPDATE_FREQ = 2500  # 10_000
LR = 2.5e-4
SAVE_PATH = ""
SAVE_INTERVAL = 5000  # 30_000
LOG_DIR = ""
LOG_INTERVAL = 1000  # 10_000
LOAD_MODEL_PARAMS = False
LOAD_PATH = "./train_params/test_mid_training.pt"
PER_BETA = 0.2
PER_BUFFER_CAPACITY = 2**14
PER_ALPHA = 0.6


def train(fg_per=True):
    """
    Function to train the agent to play in the environment

    Arguments
    ---------
        fg_per: bool
            Prioritized Experience Replay Buffer
    """
    summary_writer = SummaryWriter(LOG_DIR)

    env = make_env(ENV_ID)
    if fg_per:
        per_buffer = PrioritizedReplayBuffer(
            capacity=PER_BUFFER_CAPACITY, alpha=PER_ALPHA
        )
    else:
        replay_buffer = make_replay_buffer(env, MIN_REPLAY_SIZE, BUFFER_SIZE)

    online_net = Network(env, gamma=GAMMA, device=device).to(device)
    target_net = Network(env, gamma=GAMMA, device=device).to(device)

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Load pretrained model
    if LOAD_MODEL_PARAMS:
        print("Loading")
        prev_state = load_state(LOAD_PATH)
        online_net.load_state_dict(prev_state["state_dict"])
        optimizer.load_state_dict(prev_state["optimizer"])
        episode_count = prev_state["episode"]
        epinfos_buffer = prev_state["epinfos"]
    else:
        epinfos_buffer = deque([], maxlen=100)
        episode_count = 0

    target_net.load_state_dict(online_net.state_dict())

    obs = env.reset()[0]
    fg_watch = False
    beginning_round = True
    lives = 5

    for step in count():

        if fg_per:
            if per_buffer.binary_full():
                epsilon = np.interp(
                    step - PER_BUFFER_CAPACITY,
                    [0, EPSILON_DECAY],
                    [EPSILON_START, EPSILON_END],
                )
            else:
                epsilon = EPSILON_START
        else:
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        action = online_net.act(obs, epsilon, fg_watch)

        # Beginning of a play (speed up learning)
        if beginning_round:
            action = 1
            beginning_round = False

        new_obs, rew, done, _, info = env.step(action)

        # End of a play
        if lives != info["lives"]:
            beginning_round = True
            lives = info["lives"]

        if fg_per:
            per_buffer.add(obs, action, rew, done, new_obs)
        else:
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

        # End of an Episode
        if done:
            epinfos_buffer.append(info["episode"])
            episode_count += 1
            obs = env.reset()[0]
            beginning_round = True
            lives = 5

        obs = new_obs

        if per_buffer.binary_full():
            # Start Gradient Descent
            if fg_per:
                # Select the samples
                transitions = per_buffer.sample(BATCH_SIZE, PER_BETA)
                # Compute the loss and the Magnitude of the error
                td_errors, loss = online_net.compute_loss(
                    transitions, target_net, torch.Tensor(transitions["weights"])
                )
                # Update replay buffer priorities
                new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
                per_buffer.update_priorities(transitions["indexes"], new_priorities)
            else:
                # Select the samples
                transitions = random.sample(replay_buffer, BATCH_SIZE)
                # Compute the loss
                _, loss = online_net.compute_loss(transitions, target_net)

            # Gradient Descent
            optimizer.zero_grad()
            # Backward Propagation
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
            # Update the model weights
            optimizer.step()

            # Update Target Network
            if step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

            # Logging
            if step % LOG_INTERVAL == 0:
                rew_mean = np.mean([e["r"] for e in epinfos_buffer]) or 0
                len_mean = np.mean([e["l"] for e in epinfos_buffer]) or 0
                print()
                print("Step", step)
                print("Avg Rew", rew_mean)
                print("Len Ep", len_mean)
                print("Episodes", episode_count)
                print("epsilon", epsilon)

                # Tensorboard
                summary_writer.add_scalar("Avg Rew", rew_mean, global_step=step)
                summary_writer.add_scalar("Len Episodes", len_mean, global_step=step)
                summary_writer.add_scalar("Episodes", episode_count, global_step=step)

            # Save
            if step % SAVE_INTERVAL == 0 and step != 0:
                print("Saving")
                save_state(
                    SAVE_PATH, online_net, episode_count, optimizer, epinfos_buffer
                )


if __name__ == "__main__":
    train()
