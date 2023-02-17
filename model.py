import torch
import torch.nn as nn
import random
import numpy as np


class Network(nn.Module):
    """
        Deep Q Neural Network 
    """
    def __init__(self, env, gamma, device):
        """
            Create the Convolutional Neural Network

            Arguments
            ---------
                env: <gym.env>
                    Environment where the agent will evolve 
                gamma: float32
                    Discount factor 
                device: str
                    Location of the model on the disk
        """
        super().__init__()

        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.device = device

        n_input_channels = env.observation_space.shape[0]
        depths = (32, 64, 64)
        final_layer = 512

        self.cnn1 = nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1)
        self.lin1 = nn.Linear(3136, final_layer)
        self.lin2 = nn.Linear(final_layer, self.num_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # CNN
        x = self.cnn1(x)
        x = self.relu(x)

        x = self.cnn2(x)
        x = self.relu(x)

        x = self.cnn3(x)
        x = self.relu(x)

        x = self.flatten(x)

        # Classification Head
        x = self.lin1(x)
        x = self.relu(x)

        x = self.lin2(x)

        return x

    def act(self, obses, epsilon = 0, fg_watch = False):
        """
            Function for the model to act based on observation. During training, the agent
            follows an Epsilon Greedy policy.

            Arguments
            ---------
                obses: <gym.LazyFrames>
                    Frames of the environment 
                epsilon: float32
                    Hyperparameter for the epsilon greedy policy
                fg_watch: bool
                    If true, Turn off the epsilon greedy policy to only exploit
            
            Returns
            -------
                actions: int32
                    Action to play by the agent inside the environment
        """
        obses_t = torch.as_tensor(
            np.array(obses._frames), dtype=torch.float32, device=self.device
        )[None, :, :, :]
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values)
        actions = max_q_indices.detach().tolist()

        if not fg_watch:
            # Epsilon Greedy Policy
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions = random.randint(0, self.num_actions - 1)

        return actions

    def compute_loss(self, transitions, target_net):
        """
            Compute the loss between the online network and the target network

            Arguments
            ---------
                transitions: tuple(obses, actions, rews, dones,new_obses)
                    Transition between one step and another
                target_net: <Network>
                    Target Network
            
            Returns
            -------
                loss: float32
                    Loss of the model

        """
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        obses = np.stack([o._frames for o in obses])
        new_obses = np.stack([o._frames for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        rews_t = torch.as_tensor(
            rews, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        dones_t = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        new_obses_t = torch.as_tensor(
            new_obses, dtype=torch.float32, device=self.device
        )

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + self.gamma * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    