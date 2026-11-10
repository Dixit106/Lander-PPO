import gymnasium as gym
import numpy as np 

class LunarRewardShaper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_shaping = None

    def step(self, action):
        #A step in original environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        #Extraction of physical data from the observation array
        #AI dees the world as an array of 8 numbers
        x_pos = obs[0] #center is 0
        y_pos = obs[1] #ground is close to 0
        x_vel = obs[2] #Speed moving left/right
        y_vel = obs[3] #speed moving up/down
        angle = obs[4] #tilt of the rocket
        angular_vel = obs[5] #how fast it's spinning
        left_leg = obs[6] #1.0 if touching ground, 0.0 if not
        right_leg = obs[7] #same as left leg thing

        #custom penalties
        custom_reward = reward 

        #PENALTY A: Don't tilt too much, if angle very bad then penalty is heavy
        if abs(angle) > 0.5:
            custom_reward -= 2.0

        #PENALTY B: Fuel efficiency, teaching ai to save fuel
        if action == 2:
            custom_reward -= 0.5

        #REWARD A: safe touchdown, if both legs on the ground and barely moving, give it reward
        if left_leg == 1.0 and right_leg == 1.0 and abs(y_vel) < 0.1:
            custom_reward += 50.0

        return obs, custom_reward, terminated, truncated, info