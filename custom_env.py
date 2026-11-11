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
            custom_reward -= 0.1

        #PENALTY C: Stay close to the target, as x_pos is 0 at the exact center, we multiply by 2.0 to nag ai to move towards center
        custom_reward -= (abs(x_pos) * 5.0)

        #PENALTY D: Stop Drifting sideways, because it's going out of the area it's supposed to land
        custom_reward -= (abs(x_vel) * 3.0)

        #PENALTY E: SPEED LIMIT, vertical spped should be better so that there is a smooth landing not slaming the socket down.
        custom_reward -= (abs(y_vel) * 3.0)

        return obs, custom_reward, terminated, truncated, info