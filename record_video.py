import gymnasium as gym
from stable_baselines3 import PPO
import os

#Creating a folder to save MP4 files
video_folder = "landing_videos"
os.makedirs(video_folder, exist_ok=True)

# crating the environment again, using"rgb_array" instead of "human" as i want the engine to draw the frames in memory so we can save them, not just pop on screen.
base_env = gym.make("LunarLander-v3", render_mode="rgb_array")

#wraping the envi.. with video recorder. lambda function tells it to record every episode
env = gym.wrappers.RecordVideo(
    base_env,
    video_folder,
    episode_trigger=lambda x: True 
)

#loading the 500k step trained ai
model_path = "models/ppo_lunar_lander"
model = PPO.load(model_path)

print("Action! Recording AI landing attempts...")

#running 3 episodes to get a few different landings
for episode in range(3):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        #Need the best action, no random guessing
        action, _states = model.predict(obs, deterministic=True)

        #Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated 

    print(f"Video {episode + 1} recorded! Score: {total_reward:.2f}")

env.close()
print(f"Check the '{video_folder}' directory for your MP4 files!")
