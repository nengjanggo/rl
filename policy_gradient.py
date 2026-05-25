import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium.wrappers import ClipAction, NormalizeObservation, NormalizeReward
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm
from typing import Dict, Literal, List, Tuple


class PolicyGradientAgent:
    def __init__(
        self,
        env: gym.Env,
        method: Literal['reinforce', 'qac', 'td0ac'],
        hidden_dim: int,
        alpha: float,
        gamma: float,
        device: Literal['cpu', 'cuda']
    ):
        self.env: gym.Env = env
        self.method: Literal['reinforce', 'qac', 'td0ac'] = method
        self.hidden_dim: int = hidden_dim
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.device: Literal['cpu', 'cuda'] = device

        # Humanoid-v5 action space: Box(-0.4, 0.4, (17,), float32)
        self.action_space_size: int = env.action_space.shape[0]

        # Humanoid-v5 observation space: Box(-Inf, Inf, (348,), float64)
        self.obs_space_size: int = env.observation_space.shape[0]

        self.policy_mean = nn.Sequential(
            nn.Linear(self.obs_space_size, self.hidden_dim, device=self.device),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_space_size, device=self.device)
        )
        self.policy_std = nn.Sequential(
            nn.Linear(self.obs_space_size, self.hidden_dim, device=self.device),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_space_size, device=self.device),
            nn.Softplus()
        )

        if method == 'reinforce':
            self.episode_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = [] # list of (obs, action, reward), cleared at the end of every episode
        elif method == 'qac': # actor-critic based on action-value critic
            self.q_network = nn.Sequential(
                nn.Linear(self.obs_space_size + self.action_space_size, self.hidden_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1, device=self.device)
            )
            self.s_prev = None
            self.a_prev = None
            self.r_prev = None
        elif method == 'td0ac': # actor-critic based on TD(0)
            self.v_network = nn.Sequential(
                nn.Linear(self.obs_space_size, self.hidden_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1, device=self.device)
            )
            self.s_prev = None
            self.r_prev = None


    def get_action(
        self,
        obs: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor]:
        obs_tensor = torch.Tensor(obs).to(self.device)
        mean = self.policy_mean(obs_tensor)
        std = self.policy_std(obs_tensor)
        std += 1e-3
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().cpu().numpy(), log_prob


    def update_after_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
        log_prob: torch.Tensor
    ) -> float | None:
        if self.method == 'reinforce':
            self.episode_buffer.append((obs, action, reward))
            return
        elif self.method == 'qac':
            obs_tensor = torch.Tensor(obs).to(device=self.device)
            action_tensor = torch.Tensor(action).to(device=self.device)
            delta = None
            if self.s_prev is not None:
                Q_next = self.q_network(torch.cat([obs_tensor, action_tensor])).detach().item() # Q(s',a')
                if terminated:
                    Q_next = 0.0
                Q_prev = self.q_network(torch.cat([self.s_prev, self.a_prev])) # Q(s,a)
                Q_prev.backward()
                Q_prev = Q_prev.detach().item()
                delta = self.r_prev + self.gamma * Q_next - Q_prev
                self.log_prob_prev.backward()
                for param in self.policy_mean.parameters():
                    param.data += self.alpha * param.grad * Q_prev
                    param.grad = None
                for param in self.policy_std.parameters():
                    param.data += self.alpha * param.grad * Q_prev
                    param.grad = None
                for param in self.q_network.parameters():
                    param.data += self.alpha * delta * param.grad
                    param.grad = None
            self.s_prev = obs_tensor
            self.a_prev = action_tensor
            self.r_prev = reward
            self.log_prob_prev = log_prob
            if terminated:
                self.s_prev = None
                self.a_prev = None
                self.r_prev = None
            if delta is not None:
                return delta
        elif self.method == 'td0ac':
            obs_tensor = torch.Tensor(obs).to(device=self.device)
            action_tensor = torch.Tensor(action).to(device=self.device)
            delta = None
            if self.s_prev is not None:
                V_next = self.v_network(obs_tensor).detach().item() # V(s')
                if terminated:
                    V_next = 0.0
                V_prev = self.v_network(self.s_prev) # V(s)
                V_prev.backward()
                V_prev = V_prev.detach().item()
                delta = self.r_prev + self.gamma * V_next - V_prev
                self.log_prob_prev.backward()
                for param in self.policy_mean.parameters():
                    param.data += self.alpha * param.grad * delta
                    param.grad = None
                for param in self.policy_std.parameters():
                    param.data += self.alpha * param.grad * delta
                    param.grad = None
                for param in self.v_network.parameters():
                    param.data += self.alpha * delta * param.grad
                    param.grad = None
            self.s_prev = obs_tensor
            self.r_prev = reward
            self.log_prob_prev = log_prob
            if terminated:
                self.s_prev = None
                self.r_prev = None
            if delta is not None:
                return delta


    def update_after_episode(
        self,
    ) -> None:
        raise NotImplementedError


def train(
    method: Literal['reinforce', 'qac', 'td0ac'],
    hidden_dim: int,
    num_episodes: int,
    render_mode: Literal['human'] | None,
    alpha: float,
    gamma: float,
    use_wandb: bool,
    eval_episodes: int,
    device: Literal['cpu', 'cuda']
):
    if use_wandb:
        wandb.login()
        run_name = f'{method}_alp{alpha}_gam{gamma}'
        run = wandb.init(
            project='rl_implementation_InvertedPendulum-v5',
            name=run_name
        )

    env = gym.make(
        'InvertedPendulum-v5',
        render_mode='None',
        width=1080,
        height=1080
    ) # None for training
    env = NormalizeObservation(env)
    # env = NormalizeReward(env)
    env = ClipAction(env)
    # env.metadata['render_fps'] = 30
    agent = PolicyGradientAgent(env, method, hidden_dim, alpha, gamma, device)

    all_terminated = []
    all_truncated = []


    def rollout(
        agent,
        env: gym.Env
    ) -> Tuple[Dict[str, float], bool, bool]:
        obs, info = env.reset()
        all_info: Dict[str, List[float]] = {}
        episode_stat: Dict[str, float] = {}
        done = False

        while not done:
            info: Dict[str, np.float64 | np.ndarray]
            action, log_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if not all_info:
                all_info['reward'] = []
                if agent.method in ('qac', 'td0ac'):
                    all_info['delta'] = []
                for key, value in info.items():
                    if isinstance(value, np.float64):
                        all_info[key] = []

            # reward = reward.item()

            if agent.method in ('qac', 'td0ac') :
                delta = agent.update_after_step(obs, action, reward, terminated, next_obs, log_prob)
            else:
                raise NotImplementedError
            
            all_info['reward'].append(reward)
            if (agent.method in ('qac', 'td0ac')) and (delta is not None):
                all_info['delta'].append(delta)
            for key, value in info.items():
                if isinstance(value, np.float64):
                    all_info[key].append(value.item())

            done = terminated or truncated
            obs = next_obs

        # agent.update_after_episode()

        for key, value in all_info.items():
            episode_stat['len_episode'] = len(value)
            episode_stat[key] = sum(value) / len(value)
            if ('reward' not in key) and ('delta' not in key):
                episode_stat['max_' + key] = max(value)

        return episode_stat, terminated, truncated
        

    for episode in tqdm(range(num_episodes), desc='episode'):
        episode_stat, terminated, truncated = rollout(agent, env)

        if use_wandb:
            wandb.log(episode_stat, step=episode)

        all_terminated.append(float(terminated))
        all_truncated.append(float(truncated))
        if episode % eval_episodes == 0:
            # log stat
            terminated_ratio = sum(all_terminated) / len(all_terminated)
            truncated_ratio = sum(all_truncated) / len(all_truncated)
            if use_wandb:
                wandb.log(
                    {
                        'terminated_ratio': terminated_ratio,
                        'truncated_ratio': truncated_ratio
                    },
                    step=episode
                )
            all_terminated.clear()
            all_truncated.clear()

    if use_wandb:
        wandb.finish()
    env.close()

    if render_mode is not None:
        env = gym.make(
            'InvertedPendulum-v5',
            render_mode=render_mode,
            width=1080,
            height=1080
        )
        env.metadata['render_fps'] = 10
        env = NormalizeObservation(env)
        # env = NormalizeReward(env)
        env = ClipAction(env)
        agent.env = env
        for _ in range(100):
            rollout(agent, env)


if __name__ == '__main__':
    method: Literal['reinforce', 'qac', 'td0ac'] = 'qac'
    hidden_dim: int = 64
    num_episodes: int = 10000
    render_mode: Literal['human'] | None = 'human'
    alpha: float = 1e-4
    gamma: float = 1.0
    use_wandb: bool = True
    eval_episodes: int = 100 # evaluate the target policy every 'eval_episodes' episodes
    device: Literal['cpu', 'cuda'] = 'cuda'

    train(
        method,
        hidden_dim,
        num_episodes,
        render_mode,
        alpha,
        gamma,
        use_wandb,
        eval_episodes,
        device
    )