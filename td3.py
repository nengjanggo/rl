import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from gymnasium.wrappers import TimeLimit
from torch import nn
from torch.optim import AdamW
from typing import Literal, Tuple


def get_env(
    env_name: str,
    render_mode: str | None,
) -> gym.Env:
    env = gym.make(
        env_name,
        render_mode=render_mode
    )
    return env


class Scale(nn.Module):
    def __init__(
        self,
        scale: float
    ):
        super().__init__()
        self.scale = scale


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x * self.scale


class ReplayBuffer():
    def __init__(
        self,
        batch_size: int,
        replay_memory_size: int,
        obs_space_size: int,
        action_space_size: int,
        device: Literal['cpu', 'cuda']
    ):
        self.obs_buffer = torch.zeros((replay_memory_size, 1, obs_space_size), dtype=torch.float, device='cpu')
        self.action_buffer = torch.zeros((replay_memory_size, action_space_size), dtype=torch.float, device='cpu')
        self.reward_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.float, device='cpu')
        self.next_obs_buffer = torch.zeros((replay_memory_size, 1, obs_space_size), dtype=torch.float, device='cpu')
        self.terminated_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.float, device='cpu')
        self.truncated_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.float, device='cpu')

        self.batch_size: int = batch_size
        self.replay_memory_size: int = replay_memory_size
        self.obs_space_size: float = obs_space_size
        self.action_space_size: float = action_space_size
        self.device: Literal['cpu', 'cuda'] = device

        self.curr_idx: int = 0
        self.full: int = 0


    def increase(
        self
    ) -> None:
        self.curr_idx = (self.curr_idx + 1) % self.replay_memory_size
        if self.full < self.replay_memory_size:
            self.full += 1

    
    def append(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool,
        truncated: bool
    ) -> None:
        self.obs_buffer[self.curr_idx] = obs
        self.action_buffer[self.curr_idx] = action
        self.reward_buffer[self.curr_idx] = reward
        self.next_obs_buffer[self.curr_idx] = next_obs
        self.terminated_buffer[self.curr_idx] = float(terminated)
        self.truncated_buffer[self.curr_idx] = float(truncated)
        self.increase()
    

    def sample_minibatch(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        terminated_batch = []

        while len(obs_batch) < self.batch_size:
            indices = torch.randint(0, self.full - 1, (self.batch_size - len(obs_batch),))
            for idx in indices.tolist():
                obs_batch.append(self.obs_buffer[idx])
                action_batch.append(self.action_buffer[idx].unsqueeze(0))
                reward_batch.append(self.reward_buffer[idx].unsqueeze(0))
                next_obs_batch.append(self.next_obs_buffer[idx])
                terminated_batch.append(self.terminated_buffer[idx].unsqueeze(0))

        obs_batch = torch.cat(obs_batch).to(self.device)
        action_batch = torch.cat(action_batch).to(self.device)
        reward_batch = torch.cat(reward_batch).to(self.device)
        next_obs_batch = torch.cat(next_obs_batch).to(self.device)
        terminated_batch = torch.cat(terminated_batch).to(self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch


class TD3Agent():
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        total_train_steps: int,
        replay_memory_size: int,
        tau: float,
        discount_factor: float,
        actor_lr: float,
        critic_lr: float,
        exploration_noise_scale: float,
        target_smoothing_noise_scale: float,
        clamp_threshold: float,
        random_action_steps: int,
        replay_start_size: int,
        device: Literal['cpu', 'cuda'],
        use_wandb: bool
    ):
        super().__init__()

        self.env: gym.Env = env
        self.batch_size: int = batch_size
        self.replay_memory_size: int = replay_memory_size
        self.tau: float = tau
        self.discount_factor: float = discount_factor
        self.actor_lr: float = actor_lr
        self.critic_lr: float = critic_lr
        self.exploration_noise_scale = exploration_noise_scale
        self.target_smoothing_noise_scale = target_smoothing_noise_scale
        self.clamp_threshold: float = clamp_threshold
        self.random_action_steps: int = random_action_steps
        self.replay_start_size: int = replay_start_size
        self.device: Literal['cpu', 'cuda'] = device
        self.use_wandb: bool = use_wandb
        self.mode: Literal['train', 'eval'] = 'train'

        self.action_space_size: int = env.action_space.shape[0]
        self.action_space_high: float = env.action_space.high[0].item()
        self.obs_space_size: int = env.observation_space.shape[0]

        self.total_train_steps: int = total_train_steps
        self.trained_steps: int = 0
        self.trained_episodes: int = 0

        self.policy_network: nn.Module = self.get_policy_network()
        self.policy_target_network: nn.Module = self.get_policy_network()
        self.policy_network.to(device)
        self.policy_target_network.to(device)
        self.q_network_1: nn.Module = self.get_q_network()
        self.q_target_network_1: nn.Module = self.get_q_network()
        self.q_network_1.to(device)
        self.q_target_network_1.to(device)
        self.q_network_2: nn.Module = self.get_q_network()
        self.q_target_network_2: nn.Module = self.get_q_network()
        self.q_network_2.to(device)
        self.q_target_network_2.to(device)
        self.sync_target_network()

        self.actor_optim: torch.optim.Optimizer = AdamW(
            self.policy_network.parameters(),
            self.actor_lr
        )
        self.critic_optim_1: torch.optim.Optimizer = AdamW(
            self.q_network_1.parameters(),
            self.critic_lr
        )
        self.critic_optim_2: torch.optim.Optimizer = AdamW(
            self.q_network_2.parameters(),
            self.critic_lr
        )

        self.replay_buffer: ReplayBuffer = ReplayBuffer(batch_size, replay_memory_size, self.obs_space_size, self.action_space_size, device)


    def get_policy_network(
        self
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.obs_space_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.action_space_size),
            nn.Tanh(),
            Scale(self.action_space_high)
        )
    

    def get_q_network(
        self
    ) -> nn.Module:
        class Q(nn.Module):
            def __init__(
                self, 
                obs_space_size: int,
                action_space_size: int
            ):
                super().__init__()
                self.linear_1 = nn.Linear(obs_space_size + action_space_size, 400)
                self.linear_2 = nn.Linear(400, 300)
                self.linear_3 = nn.Linear(300, 1)


            def forward(
                self,
                obs: torch.Tensor,
                action: torch.Tensor
            ) -> torch.Tensor:
                h = self.linear_1(torch.cat([obs, action], dim=-1))
                h = F.relu(h)
                h = self.linear_2(h)
                h = F.relu(h)
                q = self.linear_3(h)
                return q
            
        
        return Q(self.obs_space_size, self.action_space_size)


    def get_action(
        self,
        obs: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            action = self.policy_network(obs).squeeze(dim=0)
        if self.mode == 'eval':
            return action.detach().cpu().numpy()
        elif self.mode == 'train':
            noised_action = action + self.exploration_noise_scale * self.action_space_high * torch.randn_like(action)
            noised_action = torch.clamp(noised_action, min=-self.action_space_high, max=self.action_space_high)
            return noised_action.detach().cpu().numpy()
    

    def update_network(
        self
    ) -> Tuple[float | None, float | None, float | None]:
        obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch = self.replay_buffer.sample_minibatch()
        q_1: torch.Tensor = self.q_network_1(obs_batch, action_batch)
        q_2: torch.Tensor = self.q_network_2(obs_batch, action_batch)
        with torch.no_grad():
            next_action_batch = self.policy_target_network(next_obs_batch)
            noised_next_action_batch = next_action_batch + self.action_space_high * torch.clamp(self.target_smoothing_noise_scale * torch.randn_like(next_action_batch), -self.clamp_threshold, self.clamp_threshold)
            q_next_1: torch.Tensor = self.q_target_network_1(next_obs_batch, noised_next_action_batch)
            q_next_2: torch.Tensor = self.q_target_network_2(next_obs_batch, noised_next_action_batch)
            target_batch = reward_batch + self.discount_factor * torch.minimum(q_next_1, q_next_2) * (1 - terminated_batch)

        actor_loss = None

        critic_loss_1 = torch.mean((target_batch - q_1) ** 2)
        critic_loss_2 = torch.mean((target_batch - q_2) ** 2)

        self.critic_optim_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optim_1.step()
        self.critic_optim_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optim_2.step()

        if self.step % 2 == 0:
            actor_loss = -self.q_network_1(obs_batch, self.policy_network(obs_batch)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.critic_optim_1.zero_grad()

        if actor_loss:
            actor_loss = actor_loss.item()
        if critic_loss_1:
            critic_loss_1 = critic_loss_1.item()
        if critic_loss_2:
            critic_loss_2 = critic_loss_2.item()

        return actor_loss, critic_loss_1, critic_loss_2


    def sync_target_network(
        self
    ) -> None:
        self.policy_target_network.load_state_dict(self.policy_network.state_dict())
        self.q_target_network_1.load_state_dict(self.q_network_1.state_dict())
        self.q_target_network_2.load_state_dict(self.q_network_2.state_dict())


    def update_target_network(
        self
    ) -> None:
        if self.step % 2 == 0:
            with torch.no_grad():
                for param, target_param in zip(self.policy_network.parameters(), self.policy_target_network.parameters()):
                    # target_param = self.tau * param + (1 - self.tau) * target_param
                    target_param.mul_(1 - self.tau)
                    target_param.add_(self.tau * param)
                for param, target_param in zip(self.q_network_1.parameters(), self.q_target_network_1.parameters()):
                    # target_param = self.tau * param + (1 - self.tau) * target_param
                    target_param.mul_(1 - self.tau)
                    target_param.add_(self.tau * param)
                for param, target_param in zip(self.q_network_2.parameters(), self.q_target_network_2.parameters()):
                    # target_param = self.tau * param + (1 - self.tau) * target_param
                    target_param.mul_(1 - self.tau)
                    target_param.add_(self.tau * param)


    def update_after_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray
    ) -> Tuple[float | None, float | None, float | None]:
        self.replay_buffer.append(
            torch.tensor(obs),
            torch.tensor(action),
            reward,
            torch.tensor(next_obs),
            terminated,
            truncated
        )

        actor_loss = None
        critic_loss_1 = None
        critic_loss_2 = None

        if self.replay_buffer.full >= self.replay_start_size:
            actor_loss, critic_loss_1, critic_loss_2 = self.update_network()
            self.update_target_network()
        self.trained_steps += 1
        return actor_loss, critic_loss_1, critic_loss_2


def train(
    env_name: str,
    agent: TD3Agent,
    use_wandb: bool,
):
    if use_wandb:
        wandb.login()
        project_name = env_name.replace('/', '-')
        run_name = 'td3'
        run = wandb.init(
            project=f'{project_name}',
            name=run_name
        )

    def rollout(
        agent: TD3Agent
    ) -> Tuple[bool, bool, float]:
        obs, info = agent.env.reset()
        done = False
        agent.step = 0
        reward_sum = 0.0
        all_actor_loss = []
        all_critic_loss_1 = []
        all_critic_loss_2 = []
        while not done:
            if agent.trained_steps < agent.random_action_steps:
                action = agent.env.action_space.sample()
            else:
                action = agent.get_action(torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(agent.device))
            next_obs, reward, terminated, truncated, info = agent.env.step(action)
            if agent.mode == 'train':
                actor_loss, critic_loss_1, critic_loss_2= agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            done = terminated or truncated
            obs = next_obs
            reward_sum += reward
            agent.step += 1

            if actor_loss is not None:
                all_actor_loss.append(actor_loss)
            if critic_loss_1 is not None:
                all_critic_loss_1.append(critic_loss_1)
            if critic_loss_2 is not None:
                all_critic_loss_2.append(critic_loss_2)

        if use_wandb:
            if all_actor_loss:
                wandb.log({'actor_loss': sum(all_actor_loss) / len(all_actor_loss)}, step=agent.trained_steps)
            if all_critic_loss_1:
                wandb.log({'critic_loss_1': sum(all_critic_loss_1) / len(all_critic_loss_1)}, step=agent.trained_steps)
            if all_critic_loss_2:
                wandb.log({'critic_loss_2': sum(all_critic_loss_2) / len(all_critic_loss_2)}, step=agent.trained_steps)

        return terminated, truncated, reward_sum
        
    while agent.trained_steps < agent.total_train_steps:
        terminated, truncated, episode_reward = rollout(agent)
        agent.trained_episodes += 1
        if use_wandb:
            wandb.log(
                {
                    'episode_reward': episode_reward,
                    'trained_episodes': agent.trained_episodes
                },
                step=agent.trained_steps
            )

    if use_wandb:
        wandb.finish()

    agent.env.close()


if __name__ == '__main__':
    batch_size: int = 100
    total_train_steps: int = 1000000
    replay_memory_size = 1000000
    tau: float = 0.005
    discount_factor: float = 0.99
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    exploration_noise_scale: float = 0.1
    target_smoothing_noise_scale: float = 0.2
    clamp_threshold: float = 0.5
    random_action_steps: int = 1000
    replay_start_size: int = 50000
    device: Literal['cpu', 'cuda'] = 'mps'
    use_wandb: bool = True

    env_name: Literal[
        'Ant-v5',
        'HalfCheetah-v5',
        'InvertedPendulum-v5',
        'Walker2d-v5'
    ] = 'Walker2d-v5'

    train_env = get_env(
        env_name=env_name,
        render_mode=None
    )
    # train_env = TimeLimit(train_env, max_episode_steps=10000)
    agent = TD3Agent(
        train_env,
        batch_size,
        total_train_steps,
        replay_memory_size,
        tau,
        discount_factor,
        actor_lr,
        critic_lr,
        exploration_noise_scale,
        target_smoothing_noise_scale,
        clamp_threshold,
        random_action_steps,
        replay_start_size,
        device,
        use_wandb
    )

    train(
        env_name,
        agent,
        use_wandb
    )