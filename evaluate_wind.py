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

        lander = self.env.unwrapped.lander

        # Roll the dice to see if a sudden wind microburst hits this frame
        if lander is not None and np.random.rand() < self.gust_prob:
            # Random horizontal force
            force_x = np.random.uniform(-self.gust_force, self.gust_force)
            # Slight downward downdraft
            force_y = np.random.uniform(-self.gust_force * 0.5, 0.0)

            # apply the impulse directly to the center of the rocket
            lander.ApplyForceToCenter((force_x, force_y), wake=True)

        return obs, reward, terminated, truncated, info 

def main():

    model_path = "models/ppo_lunar_lander"
    console.print(f"[bold cyan]Loading AI Agent from {model_path}...[/bold cyan]")
    model = PPO.load(model_path)

    weather_conditions = {
        "Clear Skies": {"enable_wind": False, "wind_power": 0.0, "turbulence": 0.0, "gust_prob": 0.0, "gust_force": 0.0},
        "Breezy": {"enable_wind": True, "wind_power": 10.0, "turbulence": 1.0, "gust_prob": 0.02, "gust_force": 10.0},
        "Gale Force": {"enable_wind": True, "wind_power": 15.0, "turbulence": 1.5, "gust_prob": 0.05, "gust_force": 25.0},
        "Category 5 Hurricane": {"enable_wind": True, "wind_power": 20.0, "turbulence": 2.0, "gust_prob": 0.10, "gust_force": 45.0},
    }

    table = Table(title="[bold red]Hurricane Protocol :: Disturbance Rejection Telemetry[/bold red]")
    table.add_column("Weather Tier", style="cyan")
    table.add_column("Wind Power", justify="right", style="blue")
    table.add_column("Microburst Force", justify="right", style="magenta")
    table.add_column("Avg Score", justify="right", style="green")
    table.add_column("Avg Steps", justify="right", style="white")
    table.add_column("Survival Rate", justify="right", style="yellow")

    console.print("\n[yellow]Running 20 stress-test flights per weather tier...[/yellow]")


    for condition, params in weather_conditions.items():
        # base environment with native wind and turbulence
        base_env = gym.make(
            "LunarLander-v3",
            render_mode="human",
            enable_wind=params["enable_wind"],
            wind_power=params["wind_power"],
            turbulence_power=params["turbulence"]
        )

        env = HurricaneWrapper(
            base_env,
            gust_prob=params["gust_prob"],
            gust_force=params["gust_force"]
        )

        scores = []
        steps_list = []
        landings = 0
        num_flights = 20

        for _ in range(num_flights):
            obs, info = env.reset()
            done = False 
            flight_score = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                flight_score += reward 
                steps += 1
                done = terminated or truncated 

            scores.append(flight_score)
            steps_list.append(steps)
            if flight_score >= 200:
                landings += 1

        avg_score = np.mean(scores)
        avg_steps = np.mean(steps_list)
        survival_rate = (landings / num_flights) * 100

        table.add_row(
            condition,
            str(params["wind_power"]),
            f"{params['gust_force']} N",
            f"{avg_score:.2f}",
            f"{avg_steps:.1f}",
            f"{survival_rate:.0f}%"
        )
        env.close()

    console.print(table)

if __name__ == "__main__":
    main()