import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium.wrappers import ClipAction, NormalizeObservation, NormalizeReward
from torch import nn
from torch.distributions import Normal
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, Literal, List, Tuple


def get_env(
    env_name: str,
    render_mode: str | None,
    normalize_observation: bool,
    normalize_reward: bool,
    clip_action: bool,
    max_episode_steps: int | None = None,
    render_fps: int | None = None
) -> gym.Env:
    env = gym.make(
        env_name,
        max_episode_steps,
        render_mode
    )
    env.metadata['render_fps'] = render_fps
    if normalize_observation:
        env = NormalizeObservation(env)
    if normalize_reward:
        env = NormalizeReward(env)
    if clip_action:
        env = ClipAction(env)

    return env
    

class PolicyGradientAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        method: Literal['reinforce', 'qac', 'td0ac'],
        hidden_dim: int,
        alpha: float,
        gamma: float,
        device: Literal['cpu', 'cuda'],
        use_wandb: bool
    ):
        super().__init__()

        self.env: gym.Env = env
        self.method: Literal['reinforce', 'qac', 'td0ac'] = method
        self.hidden_dim: int = hidden_dim
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.device: Literal['cpu', 'cuda'] = device
        self.use_wandb: bool = use_wandb
        self.episode: int = 0

        self.action_space_size: int = env.action_space.shape[0]
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
                nn.ELU(),
                nn.Linear(self.hidden_dim, 1, device=self.device)
            )
            self.s_prev = None
            self.a_prev = None
            self.r_prev = None
        elif method == 'td0ac': # actor-critic based on TD(0)
            self.v_network = nn.Sequential(
                nn.Linear(self.obs_space_size, self.hidden_dim, device=self.device),
                nn.ELU(),
                nn.Linear(self.hidden_dim, 1, device=self.device)
            )
            self.s_prev = None
            self.r_prev = None

        self.optim: torch.optim.Optimizer = AdamW(
            self.parameters(),
            self.alpha
        )


    def forward(
        self,
        obs_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy_mean(obs_tensor)
        std = self.policy_std(obs_tensor)
        return mean, std
    

    def get_action(
        self,
        obs: np.ndarray
    ) -> np.ndarray:
        obs_tensor = torch.Tensor(obs).to(self.device)
        mean, std = self.forward(obs_tensor)
        std += 1e-3
        dist = Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy()


    def update_after_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray
    ) -> float | None:
        if self.method == 'reinforce':
            self.episode_buffer.append((obs, action, reward))
            return
            
        obs_tensor = torch.Tensor(obs).to(device=self.device)
        action_tensor = torch.Tensor(action).to(device=self.device)
        delta = None


        def compute_log_prob(
            obs: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:
            mean, std = self.forward(obs)
            std += 1e-3
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum()
            return log_prob


        if self.s_prev is not None:
            self.optim.zero_grad()
            log_prob_prev = compute_log_prob(self.s_prev, self.a_prev)

            if self.method == 'qac':
                Q_next_detached: torch.Tensor = self.q_network(torch.cat([obs_tensor, action_tensor])).detach() # Q(s',a')
                if terminated:
                    Q_next_detached: torch.Tensor = torch.zeros((1,), device=self.device, requires_grad=False)
                Q_prev: torch.Tensor = self.q_network(torch.cat([self.s_prev, self.a_prev])) # Q(s,a)

                delta = self.r_prev + self.gamma * Q_next_detached - Q_prev
                q_network_loss = delta ** 2
                q_network_loss.backward()

                Q_prev_detached = Q_prev.detach().item()
                policy_loss = -(log_prob_prev * Q_prev_detached)
            elif self.method == 'td0ac':
                V_next_detached: torch.Tensor = self.v_network(obs_tensor).detach() # V(s')
                if terminated:
                    V_next_detached: torch.Tensor = torch.zeros((1,), device=self.device, requires_grad=False)
                V_prev: torch.Tensor = self.v_network(self.s_prev) # V(s)

                delta = self.r_prev + self.gamma * V_next_detached - V_prev
                v_network_loss = delta ** 2
                v_network_loss.backward()

                delta_detached = delta.detach().item()
                policy_loss = -(log_prob_prev * delta_detached)

            policy_loss.backward()
            self.optim.step()

        self.s_prev = obs_tensor
        self.a_prev = action_tensor
        self.r_prev = reward
        if terminated or truncated:
            self.s_prev = None
            self.a_prev = None
            self.r_prev = None
        if delta is not None:
            return delta


    def update_after_episode(
        self
    ) -> None:
        if self.method != 'reinforce':
            return
        
        G_t = 0.0
        self.optim.zero_grad()

        for obs, action, reward in reversed(self.episode_buffer):

            obs_tensor = torch.Tensor(obs).to(self.device)
            action_tensor = torch.Tensor(action).to(self.device)

            G_t = reward + self.gamma * G_t

            mean = self.policy_mean(obs_tensor)
            std = self.policy_std(obs_tensor)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action_tensor).sum()

            policy_loss = -(log_prob * G_t)
            policy_loss.backward()

        self.optim.step()
        self.episode_buffer.clear()


def train(
    env_name: str,
    train_env: gym.Env,
    eval_env: gym.Env,
    method: Literal['reinforce', 'qac', 'td0ac'],
    hidden_dim: int,
    num_episodes: int,
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
            project=f'rl_implementation_{env_name}',
            name=run_name
        )

    env = train_env
    agent = PolicyGradientAgent(env, method, hidden_dim, alpha, gamma, device, use_wandb)

    all_terminated = []
    all_truncated = []


    def rollout(
        agent: PolicyGradientAgent,
        env: gym.Env
    ) -> Tuple[Dict[str, float], bool, bool]:
        obs, info = env.reset()
        all_info: Dict[str, List[float]] = {}
        episode_stat: Dict[str, float] = {}
        done = False

        while not done:
            info: Dict[str, np.float64 | np.ndarray]
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if not all_info:
                all_info['reward'] = []
                if agent.method in ('qac', 'td0ac'):
                    all_info['delta'] = []
                for key, value in info.items():
                    if isinstance(value, np.float64):
                        all_info[key] = []

            # reward = reward.item()

            if agent.method == 'reinforce':
                agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            else:
                delta = agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            
            all_info['reward'].append(reward)
            if (agent.method in ('qac', 'td0ac')) and (delta is not None):
                all_info['delta'].append(delta)
            for key, value in info.items():
                if isinstance(value, np.float64):
                    all_info[key].append(value.item())

            done = terminated or truncated
            obs = next_obs

        agent.update_after_episode()

        for key, value in all_info.items():
            episode_stat['len_episode'] = len(value)
            episode_stat[key] = sum(value) / len(value)
            if ('reward' not in key) and ('delta' not in key):
                episode_stat['max_' + key] = max(value)
            if key == 'reward':
                episode_stat['reward_total'] = sum(value)

        return episode_stat, terminated, truncated
        

    for episode in tqdm(range(num_episodes), desc='episode'):
        agent.episode = episode
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

    if eval_env is not None:
        env = eval_env
        agent.env = env
        for _ in range(100):
            rollout(agent, env)


if __name__ == '__main__':
    method: Literal['reinforce', 'qac', 'td0ac'] = 'reinforce'
    hidden_dim: int = 32
    num_episodes: int = 10000
    render_mode: Literal['human'] | None = 'human'
    alpha: float = 1e-3
    gamma: float = 0.99
    use_wandb: bool = True
    eval_episodes: int = 100 # evaluate the target policy every 'eval_episodes' episodes
    device: Literal['cpu', 'cuda'] = 'cuda'

    env_name: Literal[
        'Ant-v5', # 1e-5
        'Humanoid-v5',
        'HumanoidStandup-v5',
        'InvertedPendulum-v5', # 1e-3
        'Pusher-v5'
    ] = 'InvertedPendulum-v5'
    normalize_observation: bool = False
    normalize_reward: bool = False
    clip_action: bool = True
    max_episode_steps: int = 10000

    train_env = get_env(
        env_name=env_name,
        render_mode=None,
        normalize_observation=normalize_observation,
        normalize_reward=normalize_reward,
        clip_action=clip_action,
        max_episode_steps=max_episode_steps
    )

    eval_env = get_env(
        env_name=env_name,
        render_mode=None,
        normalize_observation=normalize_observation,
        normalize_reward=normalize_reward,
        clip_action=clip_action,
        max_episode_steps=max_episode_steps,
        render_fps=10
    )

    train(
        env_name,
        train_env,
        eval_env,
        method,
        hidden_dim,
        num_episodes,
        alpha,
        gamma,
        use_wandb,
        eval_episodes,
        device
    )