import gymnasium as gym
import numpy as np
import wandb
from collections import defaultdict
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm
from typing import Literal, Tuple


class SarsaAgent:
    def __init__(
        self,
        env: gym.Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        use_expected_sarsa: bool
    ):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.use_expected_sarsa = use_expected_sarsa
        self.action_space_size = int(env.action_space.n)
        self.S = None # S for (S,A,R,S',A') update
        self.A = None # A for (S,A,R,S',A') update
        self.R = None # R for (S,A,R,S',A') update
        self.Q = defaultdict(lambda: np.zeros((self.action_space_size)))

    
    def get_action(
        self,
        obs: int
    ) -> int:
        # epsilon-greedy
        if np.random.rand() < self.epsilon: # random
            action = np.random.choice(self.action_space_size)
        else: # greedy
            prob = self.Q[obs]
            action = int(np.argmax(prob))
        
        return action
    

    def update(
        self,
        obs: int, # S'
        action: int, # A'
        reward: int, # R'
        terminated: bool, 
        next_obs: int # A''
    ) -> None:
        if self.S is not None:
            if not self.use_expected_sarsa:
                self.Q[self.S][self.A] += self.alpha * (self.R + self.gamma * self.Q[obs][action] - self.Q[self.S][self.A])
            if self.use_expected_sarsa:
                Q_next = self.Q[obs] # Q(S_{t+1},a) for all a
                pi_next = np.full((self.action_space_size), 1 / self.action_space_size)
                pi_next[Q_next.argmax()] = 1 - self.epsilon + 1 / self.action_space_size # \pi(a|S_{t+1}) for all a
                expectation = float(np.sum(Q_next * pi_next)) # E_\pi[ Q(S_{t+1},A_{t+1}) | S_{t+1} ]
                self.Q[self.S][self.A] += self.alpha * (self.R + self.gamma * expectation - self.Q[self.S][self.A])
        self.S = obs
        self.A = action
        self.R = reward
        if terminated:
            self.S = None
            self.A = None
            self.R = None


class QlearningAgent:
    def __init__(
        self,
        env: gym.Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        use_double_learning: bool
    ):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.use_double_learning = use_double_learning
        self.action_space_size = int(env.action_space.n)
        if not use_double_learning:
            self.Q = defaultdict(lambda: np.zeros((self.action_space_size)))
        if use_double_learning:
            self.Q1 = defaultdict(lambda: np.zeros((self.action_space_size)))
            self.Q2 = defaultdict(lambda: np.zeros((self.action_space_size)))

    
    def get_action(
        self,
        obs: int
    ) -> int:
        # epsilon-greedy
        if np.random.rand() < self.epsilon: # random
            action = np.random.choice(self.action_space_size)
        else: # greedy
            if not self.use_double_learning:
                dist = self.Q[obs]
            if self.use_double_learning:
                dist = (self.Q1[obs] + self.Q2[obs]) / 2
            action = np.argmax(dist)
        
        return action
    

    def update(
        self,
        obs: int, # S
        action: int, # A
        reward: int, # R
        terminated: bool, 
        next_obs: int # S'
    ) -> None:
        if not self.use_double_learning:
            self.Q[obs][action] += self.alpha * (reward + self.gamma * float(np.max(self.Q[next_obs])) - self.Q[obs][action])
        if self.use_double_learning:
            if np.random.rand() < 0.5:
                Q_left, Q_right = self.Q1, self.Q2
            else:
                Q_left, Q_right = self.Q2, self.Q1
            Q_left[obs][action] += self.alpha * (reward + self.gamma * float(Q_right[next_obs][np.argmax(Q_left[next_obs])]) - Q_left[obs][action])
        

def train(
    method: Literal['sarsa', 'expected_sarsa', 'q_learning'],
    num_episodes: int,
    render_mode: Literal['human'] | None,
    alpha: float,
    epsilon: float,
    gamma: float,
    use_double_learning: bool,
    use_wandb: bool,
    eval_episodes: int,
    eval_rollouts: int
):
    if use_wandb:
        wandb.login()
        run_name = f'{method}_alp{alpha}_eps{epsilon}_gam{gamma}'
        if use_double_learning:
            run_name = 'double_' + run_name
        run = wandb.init(
            project='rl_implementation',
            name=run_name,
            config={
                'num_episode': num_episodes
            },
        )

    env = gym.make('Taxi-v4', is_rainy=False, render_mode=None) # None for training
    wrapped_env = TimeLimit(env, max_episode_steps=200)

    if method != 'q_learning' and use_double_learning:
        raise NotImplementedError
    elif method == 'q_learning':
        agent = QlearningAgent(wrapped_env, alpha, epsilon, gamma, use_double_learning)
    else:
        use_expected_sarsa = True if method == 'expected_sarsa' else False
        agent = SarsaAgent(wrapped_env, alpha, epsilon, gamma, use_expected_sarsa)

    all_terminated = []
    all_truncated = []

    def rollout(
        agent,
        env: gym.Env
    ) -> Tuple[float, float, bool, bool]:
        obs, info = env.reset()
        all_reward = []
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                assert reward == 20, f'reward {reward}'
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs
            all_reward.append(reward)

        avg_reward = sum(all_reward) / len(all_reward)
        len_episode = len(all_reward)

        return avg_reward, len_episode, terminated, truncated
        
    for episode in tqdm(range(num_episodes), desc='episode'):
        avg_reward, len_episode, terminated, truncated = rollout(agent, wrapped_env)

        if use_wandb:
            wandb.log(
                {
                    'avg_reward': avg_reward,
                    'len_episode': len_episode
                },
                step=episode
            )

        all_terminated.append(float(terminated))
        all_truncated.append(float(truncated))
        if episode % eval_episodes == 0:
            # eval target policy
            agent.epsilon = 0.0

            reward_sum = 0.0
            len_episode_sum = 0
            target_terminated_ratio = 0.0
            target_truncated_ratio = 0.0
            for eval_episode in range(eval_rollouts):
                avg_reward, len_episode, terminated, truncated = rollout(agent, wrapped_env)
                reward_sum += avg_reward * len_episode
                len_episode_sum += len_episode
                target_terminated_ratio += (float(terminated) - target_terminated_ratio) / (eval_episode + 1)
                target_truncated_ratio += (float(truncated) - target_truncated_ratio) / (eval_episode + 1)
            target_avg_reward = reward_sum / len_episode_sum
            target_len_episode = len_episode_sum / eval_rollouts

            agent.epsilon = epsilon

            # log stat
            terminated_ratio = sum(all_terminated) / len(all_terminated)
            truncated_ratio = sum(all_truncated) / len(all_truncated)
            if use_wandb:
                wandb.log(
                    {
                        'terminated_ratio': terminated_ratio,
                        'truncated_ratio': truncated_ratio,
                        'target_avg_reward': target_avg_reward,
                        'target_len_episode': target_len_episode,
                        'target_terminated_ratio': target_terminated_ratio,
                        'target_truncated_ratio': target_truncated_ratio
                    },
                    step=episode
                )
            all_terminated.clear()
            all_truncated.clear()

    if use_wandb:
        wandb.finish()
    env.close()

    if render_mode is not None:
        env = gym.make('Taxi-v4', is_rainy=False, render_mode=render_mode)
        env.metadata['render_fps'] = 10
        wrapped_env = TimeLimit(env, max_episode_steps=200)
        agent.env = wrapped_env
        agent.epsilon = 0.0 # target policy
        for _ in range(10):
            rollout(agent, wrapped_env)


if __name__ == '__main__':
    method: Literal['sarsa', 'expected_sarsa', 'q_learning'] = 'expected_sarsa'
    num_episodes: int = 1000
    render_mode: Literal['human'] | None = 'human'
    alpha: float = 0.1
    epsilon: float = 0.1
    gamma: float = 1.0
    use_double_learning: bool = False
    use_wandb: bool = True
    eval_episodes: int = 100 # evaluate the target policy every 'eval_episodes' episodes
    eval_rollouts: int = 50 # run 'eval_rollouts' rollouts of the target policy per evaluation

    train(
        method,
        num_episodes,
        render_mode,
        alpha,
        epsilon,
        gamma,
        use_double_learning,
        use_wandb,
        eval_episodes,
        eval_rollouts
    )