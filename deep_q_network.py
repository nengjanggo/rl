import ale_py
import gymnasium as gym
import numpy as np
import torch
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
        render_mode=render_mode,
        mode=0,
        difficulty=0,
        obs_type='grayscale'
    )
    return env


class ReplayBuffer():
    def __init__(
        self,
        batch_size: int,
        replay_memory_size: int,
        device: Literal['cpu', 'cuda']
    ):
        self.obs_buffer = torch.zeros((replay_memory_size, 1, 84, 84), dtype=torch.uint8, device='cpu')
        self.action_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.int, device='cpu')
        self.reward_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.int, device='cpu')
        self.terminated_buffer = torch.zeros((replay_memory_size, 1), dtype=torch.int, device='cpu')

        self.batch_size: int = batch_size
        self.replay_memory_size: int = replay_memory_size
        self.device = device

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
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool
    ) -> None:
        assert obs.shape == (1, 1, 84, 84)
        assert next_obs.shape == (1, 1, 84, 84)

        if self.full == 0:
            self.obs_buffer[self.curr_idx] = obs
        elif int(self.terminated_buffer[self.curr_idx - 1]) == 1:
            # start of episode, now self.obs_buffer[self.curr_idx] is terminal state, so we increase self.curr_idx
            self.terminated_buffer[self.curr_idx] = terminated
            self.increase()
            self.obs_buffer[self.curr_idx] = obs
        self.action_buffer[self.curr_idx] = action
        self.reward_buffer[self.curr_idx] = reward
        self.obs_buffer[(self.curr_idx + 1) % self.replay_memory_size] = next_obs
        self.terminated_buffer[self.curr_idx] = terminated
        self.increase()
    

    def sample_minibatch(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_stack_batch = []
        action_batch = []
        reward_batch = []
        next_obs_stack_batch = []
        terminated_batch = []

        while len(obs_stack_batch) < self.batch_size:
            indices = torch.randint(0, self.full - 4, (self.batch_size - len(obs_stack_batch),))
            
            for idx in indices.tolist():
                if torch.any(self.terminated_buffer[idx : idx + 3] == 1):
                    continue
                obs_stack_batch.append(self.obs_buffer[idx : idx + 4].transpose(0, 1))
                action_batch.append(self.action_buffer[idx + 3])
                reward_batch.append(self.reward_buffer[idx + 3])
                next_obs_stack_batch.append(self.obs_buffer[idx + 1 : idx + 5].transpose(0, 1))
                terminated_batch.append(self.terminated_buffer[idx + 3])

        obs_stack_batch = torch.cat(obs_stack_batch).to(torch.float).to(self.device) / 255.0
        action_batch = torch.cat(action_batch).to(torch.int).to(self.device)
        reward_batch = torch.cat(reward_batch).to(torch.float).to(self.device)
        next_obs_stack_batch = torch.cat(next_obs_stack_batch).to(torch.float).to(self.device) / 255.0
        terminated_batch = torch.cat(terminated_batch).to(torch.float).to(self.device)

        return obs_stack_batch, action_batch, reward_batch, next_obs_stack_batch, terminated_batch
    

    def stack_recent_frames(
        self
    ) -> torch.Tensor:
        offsets = torch.arange(-3, 1)
        indices = (self.curr_idx + offsets) % self.replay_memory_size
        frame_stack = self.obs_buffer[indices]
        return frame_stack.transpose(0, 1).to(torch.float).to(self.device) / 255.0


class DQNAgent():
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        replay_memory_size: int,
        target_network_update_frequency: int,
        use_dueling: bool,
        use_double: bool,
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
        self.use_dueling: bool = use_dueling
        self.use_double: bool = use_double
        self.discount_factor: float = discount_factor
        self.lr: float = lr
        self.replay_start_size: int = replay_start_size
        self.device: Literal['cpu', 'cuda'] = device
        self.use_wandb: bool = use_wandb

        self.action_space_size: int = int(env.action_space.n) # 6
        self.obs_space_size: int = env.observation_space.shape[0] # 210

        # We trained for a total of 10 million frames and used a replay memory of one million most recent frames. - from dqn paper
        self.total_frames: int = 10 ** 7
        self.trained_frames: int = 0
        self.trained_episodes: int = 0
        self.epsilon: float = 1.0 # ... annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter. - from dqn paper
        self.epsilon_backup: float | None = None

        self.q_network: nn.Module = self.get_q_network(use_dueling)
        self.q_target_network: nn.Module = self.get_q_network(use_dueling)
        self.q_network.to(device)
        self.q_target_network.to(device)
        self.update_target_network()

        self.optim: torch.optim.Optimizer = AdamW(
            self.q_network.parameters(),
            self.lr
        )

        self.replay_buffer: ReplayBuffer = ReplayBuffer(batch_size, replay_memory_size, device)


    def train_mode(
        self
    ) -> None:
        assert self.epsilon == 0.0
        self.epsilon = self.epsilon_backup
        self.epsilon_backup = None


    def eval_mode(
        self
    ) -> None:
        assert self.epsilon_backup is None
        self.epsilon_backup = self.epsilon
        self.epsilon = 0.0


    def get_q_network(
        self,
        use_dueling: bool = False
    ) -> nn.Module:
        if not use_dueling:
            return nn.Sequential(
                nn.Conv2d(4, 16, 8, 4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2592, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_space_size)
            )
        else:
            class DuelingDQN(nn.Module):
                def __init__(self, action_space_size):
                    super().__init__()
                    self.backbone = nn.Sequential(
                        nn.Conv2d(4, 16, 8, 4),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, 4, 2),
                        nn.ReLU(),
                        nn.Flatten()
                    )
                    self.value_head = nn.Sequential(
                        nn.Linear(2592, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )                
                    self.advantage_head = nn.Sequential(
                        nn.Linear(2592, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_space_size)
                    )


                def forward(
                    self,
                    x: torch.Tensor
                ):
                    latent: torch.Tensor = self.backbone(x)
                    value: torch.Tensor = self.value_head(latent)
                    advantage: torch.Tensor = self.advantage_head(latent)
                    return value + advantage - advantage.mean(dim=-1, keepdim=True)


            return DuelingDQN(self.action_space_size)

    
    def preprocess_obs(
        self,
        obs: np.ndarray # (210,160)
    ) -> torch.Tensor:
        obs_tensor = torch.Tensor(obs).unsqueeze(0).unsqueeze(0) # (1,1,210,160)
        preprocessed_obs = nn.functional.interpolate(obs_tensor, (110, 84))[:,:,18:102,:].to(torch.uint8) # (1,1,84,84)
        return preprocessed_obs


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
    

    def update_network(
        self
    ) -> Tuple[float, float]:
        obs_stack_batch, action_batch, reward_batch, next_obs_stack_batch, terminated_batch = self.replay_buffer.sample_minibatch()
        q: torch.Tensor = self.q_network(obs_stack_batch)[torch.arange(self.batch_size), action_batch]
        with torch.no_grad():
            q_next: torch.Tensor = self.q_target_network(next_obs_stack_batch)
            if not self.use_double:
                target_batch = reward_batch + self.discount_factor * q_next.max(dim=-1).values * (1 - terminated_batch)
            elif self.use_double:
                argmax_action_batch: torch.Tensor = self.q_network(next_obs_stack_batch).argmax(dim=-1)
                target_batch = reward_batch + self.discount_factor * q_next[torch.arange(self.batch_size), argmax_action_batch] * (1 - terminated_batch)
        loss = torch.sum((target_batch - q) ** 2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), q.mean().item()


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
    ) -> Tuple[float, float] | Tuple[None, None]:
        self.replay_buffer.append(
            self.preprocess_obs(obs),
            action,
            reward,
            self.preprocess_obs(next_obs),
            terminated
        )

        q_network_loss = None
        q_mean = None
        if self.replay_buffer.full >= self.replay_start_size:
            q_network_loss, q_mean = self.update_network()
        if self.trained_frames % self.target_network_update_frequency == 0:
            self.update_target_network()
        self.trained_frames += 1
        self.decay_epsilon()
        return q_network_loss, q_mean


def train(
    env_name: str,
    agent: DQNAgent,
    eval_env: gym.Env,
    use_wandb: bool,
):
    if use_wandb:
        wandb.login()
        project_name = env_name.replace('/', '-')
        run_name = 'run'
        if agent.use_double:
            run_name += 'Double'
        if agent.use_dueling:
            run_name += 'Dueling'
        run = wandb.init(
            project=f'{project_name}',
            name=run_name
        )

    def rollout(
        agent: DQNAgent
    ) -> Tuple[bool, bool, float]:
        obs, info = agent.env.reset()
        done = False
        step = 0
        reward_sum = 0.0
        all_q_network_loss = []
        all_q_mean = []
        while not done:
            if step <= 4:
                action = 0 # noop
            else:
                stacked_frames = agent.replay_buffer.stack_recent_frames()
                action = agent.get_action(stacked_frames)
            next_obs, reward, terminated, truncated, info = agent.env.step(action)
            if agent.epsilon_backup is None:
                if reward == 1.0 or reward == -1.0:
                    step = 0
                    q_network_loss, q_mean= agent.update_after_step(obs, action, reward, True, truncated, next_obs)
                else:
                    q_network_loss, q_mean= agent.update_after_step(obs, action, reward, terminated, truncated, next_obs)
            done = terminated or truncated
            obs = next_obs
            reward_sum += reward
            step += 1

            if q_network_loss:
                all_q_network_loss.append(q_network_loss)
                all_q_mean.append(q_mean)

        if all_q_network_loss and use_wandb:
            wandb.log(
                {
                    'q_network_loss': sum(all_q_network_loss) / len(all_q_network_loss),
                    'q_mean': sum(all_q_mean) / len(all_q_mean)
                },
                step=agent.trained_frames
            )


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
        agent.eval_mode()
        for _ in range(100):
            rollout(agent)


if __name__ == '__main__':
    batch_size: int = 32
    replay_memory_size = 1000000
    target_network_update_frequency: int = 10000
    use_dueling: bool = True
    use_double: bool = True
    discount_factor: float = 0.99
    lr: float = 3e-4
    replay_start_size: int = 50000
    # replay_start_size: int = 5000

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
        use_dueling,
        use_double,
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