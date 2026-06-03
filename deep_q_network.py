import ale_py
import gymnasium as gym
import numpy as np
import random
import torch
import wandb
from collections import deque
from itertools import islice
from torch import nn
from torch.optim import AdamW
from typing import Dict, Deque, Literal, Tuple


def get_env(
    env_name: str,
    render_mode: str | None,
) -> gym.Env:
    env = gym.make(
        env_name,
        render_mode=render_mode,
        mode=0,
        difficulty=0,
        obs_type='grayscale'
    )
    return env


class DQNAgent():
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        replay_memory_size: int,
        target_network_update_frequency: int,
        discount_factor: float,
        lr: float,
        replay_start_size: int,
        device: Literal['cpu', 'cuda'],
        use_wandb: bool
    ):
        super().__init__()

        self.env: gym.Env = env
        self.batch_size: int = batch_size
        self.replay_memory_size: int = replay_memory_size
        self.target_network_update_frequency: int = target_network_update_frequency
        self.discount_factor: float = discount_factor
        self.lr: float = lr
        self.replay_start_size: int = replay_start_size
        self.device: Literal['cpu', 'cuda'] = device
        self.use_wandb: bool = use_wandb

        self.action_space_size: int = int(env.action_space.n) # 6
        self.obs_space_size: int = env.observation_space.shape[0] # 210

        # We trained for a total of 10 million frames and used a replay memory of one million most recent frames. - from dqn paper
        self.total_frames = 10 ** 7
        self.trained_frames = 0
        self.trained_episodes = 0
        self.epsilon = 1.0 # ... annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter. - from dqn paper
        self.replay_buffer: Deque[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=replay_memory_size)

        self.q_network: nn.Module = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space_size)
        )
        self.q_target_network: nn.Module = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space_size)
        )
        self.q_network.to(device)
        self.q_target_network.to(device)
        self.update_target_network()

        self.optim: torch.optim.Optimizer = AdamW(
            self.q_network.parameters(),
            self.lr
        )

    
    def preprocess_obs(
        self,
        obs: np.ndarray # (210,160)
    ) -> torch.Tensor:
        obs_tensor = torch.Tensor(obs).unsqueeze(0).unsqueeze(0) # (1,1,210,160)
        preprocessed_obs = nn.functional.interpolate(obs_tensor, (110, 84))[:,:,18:102,:] # (1,1,84,84)
        return preprocessed_obs
    

    def stack_frames(
        self
    ) -> torch.Tensor:
        frames = [self.replay_buffer[-4][-2], self.replay_buffer[-3][-2], self.replay_buffer[-2][-2], self.replay_buffer[-1][-2]]
        stacked_frames = torch.concatenate(frames, dim=1)
        return stacked_frames


    def get_action(
        self,
        stacked_frames: torch.Tensor
    ) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            q_values = self.q_network(stacked_frames.to(self.device))
            action = int(torch.argmax(q_values))
        return action
    

    def sample_minibatch(
        self
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]:
        obs_stack_batch = []
        action_batch = []
        reward_batch = []
        next_obs_stack_batch = []
        terminated_batch = []

        num_collected_samples = 0
        while num_collected_samples < self.batch_size:
            indices = random.sample(range(len(self.replay_buffer) - 3), self.batch_size - num_collected_samples)
            for idx in indices:
                slice = list(islice(self.replay_buffer, idx, idx + 4))
                if True in [tuple[-1] for tuple in slice[:-1]]: # tuple[-1]: terminated
                    continue
                obs_stack = torch.concatenate([tuple[0] for tuple in slice], dim=1)
                action = slice[-1][1]
                reward = slice[-1][2]
                next_obs_stack = torch.concatenate([tuple[-2] for tuple in slice], dim=1)
                terminated = slice[-1][-1]

                obs_stack_batch.append(obs_stack)
                action_batch.append(action)
                reward_batch.append(reward)
                next_obs_stack_batch.append(next_obs_stack)
                terminated_batch.append(float(terminated))

                num_collected_samples += 1

        obs_stack_batch = torch.concatenate(obs_stack_batch, dim=0).to(self.device)
        action_batch = torch.Tensor(action_batch).to(torch.int).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        next_obs_stack_batch = torch.concatenate(next_obs_stack_batch, dim=0).to(self.device)
        terminated_batch = torch.Tensor(terminated_batch).to(self.device)

        return obs_stack_batch, action_batch, reward_batch, next_obs_stack_batch, terminated_batch
    

    def update_network(
        self
    ) -> None:
        obs_stack_batch, action_batch, reward_batch, next_obs_stack_batch, terminated_batch = self.sample_minibatch()
        q: torch.Tensor = self.q_network(obs_stack_batch.to(self.device))[torch.arange(self.batch_size), action_batch]
        with torch.no_grad():
            q_next: torch.Tensor = self.q_target_network(next_obs_stack_batch.to(self.device))
            target_batch = reward_batch + self.discount_factor * q_next.max(dim=-1).values * (1 - terminated_batch)
        loss = torch.mean((target_batch - q) ** 2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def update_target_network(
        self
    ) -> None:
        self.q_target_network.load_state_dict(self.q_network.state_dict())


    # todo: replace hardcoded values with attributes
    def decay_epsilon(
        self
    ) -> None:
        if self.trained_frames > 10 ** 6:
            return
        self.epsilon -= (1.0 - 0.1) / (10 ** 6)


    def update_after_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray
    ) -> None:
        self.replay_buffer.append(
            (
                self.preprocess_obs(obs),
                action,
                reward,
                self.preprocess_obs(next_obs),
                terminated
            )
        )

        if len(self.replay_buffer) >= self.replay_start_size:
            self.update_network()
        if self.trained_frames % self.target_network_update_frequency == 0:
            self.update_target_network()
        self.trained_frames += 1
        self.decay_epsilon()


def train(
    env_name: str,
    agent: DQNAgent,
    eval_env: gym.Env,
    use_wandb: bool,
):
    if use_wandb:
        wandb.login()
        run_name = env_name.replace('/', '-')
        run = wandb.init(
            project=f'rl_implementation_{run_name}',
            name=run_name
        )

    def rollout(
        agent: DQNAgent
    ) -> Tuple[bool, bool, float]:
        obs, info = agent.env.reset()
        done = False
        step = 0
        reward_sum = 0.0
        while not done:
            if step <= 4:
                action = 0 # noop
            elif step % 4 == 0:
                stacked_frames = agent.stack_frames()
                action = agent.get_action(stacked_frames)
            else:
                pass # repeat action
            next_obs, reward, terminated, truncated, info = agent.env.step(action)
            agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            done = terminated or truncated
            obs = next_obs
            reward_sum += reward
            step += 1

        return terminated, truncated, reward_sum
        
    while agent.trained_frames < agent.total_frames:
        terminated, truncated, episode_reward = rollout(agent)
        agent.trained_episodes += 1
        if use_wandb:
            wandb.log(
                {
                    'episode_reward': episode_reward,
                    'trained_episodes': agent.trained_episodes,
                    'epsilon': agent.epsilon
                },
                step=agent.trained_frames
            )

    if use_wandb:
        wandb.finish()

    agent.env.close()

    if eval_env is not None:
        agent.env = eval_env
        for _ in range(100):
            rollout(agent)


if __name__ == '__main__':
    batch_size: int = 32
    replay_memory_size = 1000000
    target_network_update_frequency: int = 10000
    discount_factor: float = 0.99
    lr: float = 1e-3
    # replay_start_size: int = 50000
    replay_start_size: int = 500

    use_wandb: bool = True
    device: Literal['cpu', 'cuda'] = 'cuda'

    env_name: Literal[
        'ALE/Pong-v5'
    ] = 'ALE/Pong-v5'

    train_env = get_env(
        env_name=env_name,
        render_mode=None
    )
    eval_env = get_env(
        env_name=env_name,
        render_mode='human'
    )
    agent = DQNAgent(
        train_env,
        batch_size,
        replay_memory_size,
        target_network_update_frequency,
        discount_factor,
        lr,
        replay_start_size,
        device,
        use_wandb
    )

    train(
        env_name,
        agent,
        eval_env,
        use_wandb
    )