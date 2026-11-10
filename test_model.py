import gymnasium as gym
from stable_baselines3 import PPO

#creating environment again but with rendering on so that i can also see
env = gym.make("LunarLander-v3", render_mode="human")

#Now we load the trained ai we just trained from train.py
model_path = "models/ppo_lunar_lander"
model = PPO.load(model_path)

print("Testing the trained AI...")

#5 landing attempts
for episode in range(5):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        #ai predicts the best action based on it's observation with best guess no randomness
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward 

        #check if crashed, landed or ran out of time for the episode
        done = terminated or truncated 

    print(f"Episode {episode + 1} finished with a score of: {total_reward:.2f}")
env.close()