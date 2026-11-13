import gymnasium as gym
from stable_baselines3 import PPO 
import numpy as np
from rich.console import Console 
from rich.table import Table 

console = Console()

class HurricaneWrapper(gym.Wrapper):
    """
    Custom physics wrapper that injects random horizontal wind gusts and sudden vertical downdrafts directly into the Box2D lander body.
    """
    def __init__(self, env, gust_prob=0.05, gust_force=15.0):
        super().__init__(env)
        self.gust_prob = gust_prob 
        self.gust_force = gust_force 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #access raw Box2D physics body inside the gymnasium environemt
        lander = self.env.unwrapper.lander

        #Roll the dice to see if a sudden wind microburst hits this frame
        if lander is not None and np.random.rand() < self.gust_prob:
            #Random horizontal force
            force_x = np.random.uniform(-self.gust_force, self.gust_force)
            #Slight downward downdraft
            force_y = np.random.uniform(-self.gust_force * 0.5, 0.0)

            #apply th e impulse directly to the center of the rocket
            lander.ApplyForceToCenter((force_x, force_y), wake=True)

        return obs, reward, terminated, truncated, info 

    def main():
        model_path = "model/ppo_lunar_lander"
        console.print(f"[bold cyan]Loading AI Agent from {model_path}...[/bold cyan]")
        model = PPO.load(model_path)