import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="human")
#reset enviroment to start
observation, info = env.reset()

for _ in range(300):
    #agent takes a random action(0,1,2 or 3)
    #0 will be do nothing, 1 =fire left engine, 2=fire main engine, 3= fire right engine
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    #if lander crashes or lands, it reset it
    if terminated or truncated:
        observation, info = env.reset()

env.close()
