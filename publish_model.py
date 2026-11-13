from stable_baselines3 import PPO 
from huggingface_sb3 import package_to_hub 
from stable_baselines3.common.env_util import make_vec_env 

#Defining model paths and hugging face repo
model_path = "models/ppo_lunar_lander"
env_id = "LunarLander-v3"
repo_id = "Dixit106/ppo-LunarLander-v3"

print(f"Loading 3-Million Step Model from {model_path}...")
model = PPO.load(model_path)

#creating a fresh evalution environment as hugging face librabry usess this to play a round and record an mp4 video
eval_env = make_vec_env(env_id, n_envs=1)

print("Packaging model, generating video, and pushing to the Hub...")

package_to_hub(
    model=model,
    model_name="ppo-LunarLander-v3",
    model_architecture="PPO",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message="Initial release of Lander-PPO AI built by a Hackcluber"
)

print("\nDeploy complete! Check your Hugging Face profile.")