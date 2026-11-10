from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import os 
import json 

#building a function to parse the monitor files that Stable-baselines generates
def plot_training_result(log_folder, title="Lunar Lander Training Progress"):
    """
    Reads the training logs and plots the episode rewards over time.
    """
    print(f"Looking for log data in: {log_folder}")

    #will expand this later to parse the Tensorboard logs directly
    fig = plt.figure(title, figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')

    #Adding a horizontal line at 200, which is considered a "solved" environment in LunarLander
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold (200)')

    plt.grid(True, alpha=0.3)
    plt.legend()

    #Save the plot as an image file
    plt.savefig("training_curve.png")
    print("Saved training curve to training_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_training_result("./ppo_lunar_tensorboard/")