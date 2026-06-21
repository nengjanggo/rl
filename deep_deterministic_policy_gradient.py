import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
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


class DDPGAgent():
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
        initial_noise_scale: float,
        final_noise_scale: float,
        noise_scale_decay_steps: int,
        random_action_steps: int,
        actor_replay_start_size: int,
        critic_replay_start_size: int,
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
        self.noise_scale: float = initial_noise_scale
        self.initial_noise_scale: float = initial_noise_scale
        self.final_noise_scale: float = final_noise_scale
        self.noise_scale_decay_steps: int = noise_scale_decay_steps
        self.random_action_steps: int = random_action_steps
        self.actor_replay_start_size: int = actor_replay_start_size
        self.critic_replay_start_size: int = critic_replay_start_size
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
        self.q_network: nn.Module = self.get_q_network()
        self.q_target_network: nn.Module = self.get_q_network()
        self.q_network.to(device)
        self.q_target_network.to(device)
        self.sync_target_network()

        self.actor_optim: torch.optim.Optimizer = AdamW(
            self.policy_network.parameters(),
            self.actor_lr
        )
        self.critic_optim: torch.optim.Optimizer = AdamW(
            self.q_network.parameters(),
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
                self.linear_1 = nn.Linear(obs_space_size, 400)
                self.linear_2 = nn.Linear(400, 300)
                self.linear_3 = nn.Linear(300 + action_space_size, 1)


            def forward(
                self,
                obs: torch.Tensor,
                action: torch.Tensor
            ) -> torch.Tensor:
                h = self.linear_1(obs)
                h = F.relu(h)
                h = self.linear_2(h)
                h = F.relu(h)
                h = torch.cat([h, action], dim=-1)
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
            noised_action = action + self.noise_scale * self.action_space_high * torch.randn_like(action)
            noised_action = torch.clamp(noised_action, min=-self.action_space_high, max=self.action_space_high)
            return noised_action.detach().cpu().numpy()
    

    def update_network(
        self
    ) -> Tuple[float | None, float | None]:
        obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch = self.replay_buffer.sample_minibatch()
        q: torch.Tensor = self.q_network(obs_batch, action_batch)
        with torch.no_grad():
            q_next: torch.Tensor = self.q_target_network(next_obs_batch, self.policy_target_network(next_obs_batch))
            target_batch = reward_batch + self.discount_factor * q_next * (1 - terminated_batch)

        critic_loss = None
        actor_loss = None

        if self.replay_buffer.full >= self.critic_replay_start_size:
            critic_loss = torch.mean((target_batch - q) ** 2)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        if self.replay_buffer.full >= self.actor_replay_start_size:
            actor_loss = -self.q_network(obs_batch, self.policy_network(obs_batch)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.critic_optim.zero_grad()

        return actor_loss.item() if actor_loss else actor_loss, critic_loss.item() if critic_loss else critic_loss


    def sync_target_network(
        self
    ) -> None:
        self.policy_target_network.load_state_dict(self.policy_network.state_dict())
        self.q_target_network.load_state_dict(self.q_network.state_dict())


    def update_target_network(
        self
    ) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.policy_network.parameters(), self.policy_target_network.parameters()):
                # target_param = self.tau * param + (1 - self.tau) * target_param
                target_param.mul_(1 - self.tau)
                target_param.add_(self.tau * param)
            for param, target_param in zip(self.q_network.parameters(), self.q_target_network.parameters()):
                # target_param = self.tau * param + (1 - self.tau) * target_param
                target_param.mul_(1 - self.tau)
                target_param.add_(self.tau * param)


    def decay_noise_scale(
        self
    ) -> None:
        if self.trained_steps >= self.noise_scale_decay_steps:
            return
        self.noise_scale -= (self.initial_noise_scale - self.final_noise_scale) / self.noise_scale_decay_steps


    def update_after_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray
    ) -> Tuple[float | None, float | None]:
        self.replay_buffer.append(
            torch.tensor(obs),
            torch.tensor(action),
            reward,
            torch.tensor(next_obs),
            terminated,
            truncated
        )

        actor_loss = None
        critic_loss = None
        if self.replay_buffer.full >= min(self.actor_replay_start_size, self.critic_replay_start_size):
            actor_loss, critic_loss = self.update_network()
            self.update_target_network()
        self.trained_steps += 1
        self.decay_noise_scale()
        return actor_loss, critic_loss


def train(
    env_name: str,
    agent: DDPGAgent,
    use_wandb: bool,
):
    if use_wandb:
        wandb.login()
        project_name = env_name.replace('/', '-')
        run_name = 'run'
        run = wandb.init(
            project=f'{project_name}',
            name=run_name
        )

    def rollout(
        agent: DDPGAgent
    ) -> Tuple[bool, bool, float]:
        obs, info = agent.env.reset()
        done = False
        step = 0
        reward_sum = 0.0
        all_actor_loss = []
        all_critic_loss = []
        while not done:
            if agent.trained_steps < agent.random_action_steps:
                action = agent.env.action_space.sample()
            else:
                action = agent.get_action(torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(agent.device))
            next_obs, reward, terminated, truncated, info = agent.env.step(action)
            if agent.mode == 'train':
                actor_loss, critic_loss= agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            done = terminated or truncated
            obs = next_obs
            reward_sum += reward
            step += 1

            if actor_loss is not None:
                all_actor_loss.append(actor_loss)
            if critic_loss is not None:
                all_critic_loss.append(critic_loss)

        if use_wandb:
            if all_actor_loss:
                wandb.log({'actor_loss': sum(all_actor_loss) / len(all_actor_loss)}, step=agent.trained_steps)
            if all_critic_loss:
                wandb.log({'critic_loss': sum(all_critic_loss) / len(all_critic_loss)}, step=agent.trained_steps)

        return terminated, truncated, reward_sum
        
    while agent.trained_steps < agent.total_train_steps:
        terminated, truncated, episode_reward = rollout(agent)
        agent.trained_episodes += 1
        if use_wandb:
            wandb.log(
                {
                    'episode_reward': episode_reward,
                    'trained_episodes': agent.trained_episodes,
                    'noise_scale': agent.noise_scale
                },
                step=agent.trained_steps
            )

    if use_wandb:
        wandb.finish()

    agent.env.close()


if __name__ == '__main__':
    batch_size: int = 64
    total_train_steps: int = 2500000
    replay_memory_size = 1000000
    tau: float = 0.001
    discount_factor: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    initial_noise_scale: float = 0.2
    final_noise_scale: float = 0.0
    noise_scale_decay_steps: int = int(0.5 * total_train_steps)
    random_action_steps: int = 50000
    actor_replay_start_size: int = 500000
    critic_replay_start_size: int = 100000
    device: Literal['cpu', 'cuda'] = 'cuda'
    use_wandb: bool = True

    env_name: Literal[
        'Ant-v5',
        'HalfCheetah-v5',
        'InvertedPendulum-v5',
        'Walker2d-v5'
    ] = 'InvertedPendulum-v5'

    train_env = get_env(
        env_name=env_name,
        render_mode=None
    )
    agent = DDPGAgent(
        train_env,
        batch_size,
        total_train_steps,
        replay_memory_size,
        tau,
        discount_factor,
        actor_lr,
        critic_lr,
        initial_noise_scale,
        final_noise_scale,
        noise_scale_decay_steps,
        random_action_steps,
        actor_replay_start_size,
        critic_replay_start_size,
        device,
        use_wandb
    )

    train(
        env_name,
        agent,
        use_wandb
    )