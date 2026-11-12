import gymnasium as gym 
import pygame 
import time 
from rich.console import Console 

console = Console()

def main():
    console.print("[bold cyan]Initializing Project Touchdown: Manual Override...[/bold cyan]")

    # using human render
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()

    # Initialize Pygame event capture
    pygame.init()

    console.print("[bold yellow]CONTROLS:[/bold yellow]")
    console.print(" [UP ARROW] or [W]: Fire Main Engine")
    console.print(" [LEFT ARROW] or [A]: Fire Right Thruster to Push Left")
    console.print(" [RIGHT ARROW] or [D]: Fire Left Thruster to Push Right")
    console.print(" [ESC] : Quit Simulation")
    console.print("\n[bold red]Good Luck. Gravity is unforgiving.[/bold red]")

    done = False 
    total_reward = 0

    # LunarLander runs at 50 Frames Per Second. but modern CPU will run it on maybe 2000fps and crash so that needs a fix
    clock = pygame.time.Clock()

    while True:
        action = 0 # do nothing

        # Pygame window events process like click X to close etc
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return

        # keys held down for continuous engine burn
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 2 # main engine
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 3 # Right engine to push left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 1 # Left engine to push right

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward 

        # crash or successful landing 
        if terminated or truncated:
            color = "green" if total_reward >= 200 else "red"
            console.print(f"Flight ended. Final Score: [{color}]{total_reward:.2f}[/{color}]")
            time.sleep(1.5) # to process what happened
            obs, info = env.reset()
            total_reward = 0

        # Lock game loop to 50 FPS
        clock.tick(50)

if __name__ == "__main__":
    main()