import gymnasium as gym
from stable_baselines3 import PPO 
import numpy as np
from rich.console import Console 
from rich.table import Table 

#loading my best model
model_path = "models/ppo_lunar_lander"
print(f"Waking up AI from {model_path}...")
model = PPO.load(model_path)

#physics for other planets, default game gravity is roughhly Earth-like
planets = {
    "Moon": -1.62,
    "Mars": -3.72,
    "Earth (Default)": -10.0
}

#terminal UI table setup
console = Console()
table = Table(title="[bold magenta]Zero-Shot Planetary Gravity Evaluation[/bold magenta]")
table.add_column("Planet", style="cyan", no_wrap=True)
table.add_column("Gravity (m/s²)", justify="right", style="blue")
table.add_column("Avg Score", justify="right", style="green")
table.add_column("Survival Rate", justify="right", style="yellow")

print("\nCommencing interstellar flight tests. Running 10 flights per planet...\n")

#simulation loop
for planet, gravity in planets.items():
    #passing custom gravity directly into physics engine
    env = gym.make("LunarLander-v3", gravity=gravity)

    scores = []
    landings = 0

    for flight in range(10):
        obs, info = env.reset()
        done = False 
        flight_score = 0

        while not done:
            #deterministic=True, which means AI uses it's best guess, no randomness
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            flight_score += reward 
            done = terminated or truncated 

        scores.append(flight_score)
        #base game, surviving and landing yields 200+ points
        if flight_score >= 200:
            landings += 1

    avg_score = np.mean(scores)
    survival_rate = (landings / 10) * 100

    #Adding results to terminal table
    table.add_row(
        planet,
        str(gravity),
        f"{avg_score:.2f}",
        f"{survival_rate}%"
    )

    env.close()

console.print(table)