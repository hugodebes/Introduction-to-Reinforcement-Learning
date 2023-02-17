import os
import torch
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    RecordEpisodeStatistics,
)
from collections import deque
from tqdm import tqdm


def make_env(env_id, fg_watch=False):
    """
        Create a Gymnasium Environment with the Atari Preprocessing 

        Arguments
        ---------
            env_id: str
                Name of the Environment to create
            fg_watch: bool
                Training or demo (with rendering) mode
        
        Returns
        -------
            env : <gym.env>
                Environment of the Game
    """
    if fg_watch:
        env = gym.make(env_id, frameskip=1, render_mode="human")
    else:
        env = gym.make(env_id, frameskip=1)
    env = RecordEpisodeStatistics(env)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)
    return env


def make_replay_buffer(env, min_replay_size, buffer_size):
    """
        Create a Simple replay buffer for the Agent to sample it during training

        Arguments
        ---------
            env: <gym.env>
                Environment of the game
            min_replay_size: int32
                To avoid a cold start, we initialize the buffer with at least this number
                of transitions
            buffer_size: int32
                Size of the buffer (if exceeded, push the oldest one)
        
        Returns
        -------
            replay_buffer: deque
                Collection of transitions for the agent to learn 
    """
    replay_buffer = deque(maxlen=buffer_size)
    obs = env.reset()[0]
    beginning_round = True
    lives = 5
    # Min number of transitions
    for _ in tqdm(range(min_replay_size)):
        action = env.action_space.sample()

        # Start of a play (speed up training)
        if beginning_round:
            action = 1
            beginning_round = False

        new_obs, rew, done, _, info = env.step(action)

        # End of a play
        if lives != info["lives"]:
            beginning_round = True
            lives = info["lives"]

        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        obs = new_obs

        # End of an episode
        if done:
            obs = env.reset()[0]
            beginning_round = True
            lives = 5
    return replay_buffer


def load_state(load_path):
    """
        Load a state to resume the trianing

        Arguments
        ---------
            load_path: str
                Location of the file
        
        Returns
        -------
            state: dict
                State dictionary of the model, the optimizer, the number of episodes and infos
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(load_path)

    state = torch.load(load_path)
    return state


def save_state(save_path, model, episode_count, optimizer, epinfos_buffer):
    """
        Save the current training parameters

        Arguments
        ---------
            save_path: str
                Location to save the file (Warning, the folder must be created)
            model: <Network>
                Online Network 
            episode_count: int32
                Number of past episodes
            optimizer: torch.optim
                Current optimizer of the training
            epinfos_buffer: deque
                Info about the different episodes
    """
    state = {
        "episode": episode_count,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epinfos": epinfos_buffer,
    }
    torch.save(state, save_path)
