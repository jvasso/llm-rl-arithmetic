

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

fr


class RewardCallback(BaseCallback):

    def __init__(self, verbose=0, num_episodes=10):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = deque(maxlen=num_episodes)


    def _on_step(self) -> bool:
        
        self.episode_rewards.append(None)
        return True
    
    
    def get_last_train_reward(self):
        return list(self.episode_rewards)