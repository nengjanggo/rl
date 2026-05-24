from typing import Tuple, List, Literal


class Env():
    '''gridworld environment for example 4.1'''
    transition = {
        'up': (-1, 0),
        'down': (1, 0),
        'right': (0, 1),
        'left': (0, -1)
    }
    min_state = 1
    max_state = 14

    def __init__(self):
        pass

    def step(
        self,
        state: int,
        action: Literal['up', 'down', 'right', 'left']
    ) -> Tuple[int, int]:
        assert self.min_state <= state <= self.max_state, 'invalid state'
        row, col = state // 4, state % 4

        drow, dcol = self.transition[action]

        row += drow
        col += dcol

        if not (0 <= row <= 3 and 0 <= col <= 3):
            # undo
            row -= drow
            col -= dcol

        next_state = 4 * row + col
        reward = -1

        return next_state, reward
    

class Agent():
    '''agent for example 4.1'''
    def __init__(
        self, 
        env: Env
    ):
        self.env = env
        self.policy = { # stateless
            'up': 1/4,
            'down': 1/4, 
            'right': 1/4, 
            'left': 1/4
        }
        self.state_values = [0.0] * 16

    def evaluate_policy(
        self,
        inplace: bool
    ) -> List[float]:
        if inplace:
            next_state_values = self.state_values
        else:
            next_state_values = [0.0] * 16

        for state in range(self.env.min_state, self.env.max_state + 1):
            next_state_value = 0.0

            for action, prob in self.policy.items():
                next_state, reward = self.env.step(state, action)
                
                # p(s',r|s,a) = 1, \gamma = 1 
                next_state_value += prob * (reward + self.state_values[next_state])

            next_state_values[state] = next_state_value

        if not inplace:
            self.state_values = next_state_values.copy()

        return self.state_values.copy()


def iterative_policy_evaluation(
    thres: float,
    inplace: bool
):
    env = Env()
    agent = Agent(env)
    state_values = agent.state_values.copy()
    iteration_count = 0

    while True:
        next_state_values = agent.evaluate_policy(inplace)
        max_delta = 0.0

        for state in range(env.min_state, env.max_state + 1):
            delta = abs(next_state_values[state] - state_values[state])
            max_delta = max(max_delta, delta)

        state_values = next_state_values
        iteration_count += 1

        if max_delta < thres:
            break
    
    for state_value in next_state_values:
        print(f'{state_value:.2f}')

    print(f'iteration: {iteration_count}')


if __name__ == '__main__':
    iterative_policy_evaluation(0.01, False)