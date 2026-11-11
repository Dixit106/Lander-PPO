# Lander-PPO


# What is this Project in breif
**Lander-PPO**, is a project where an AI agent is trained to land a rocket using Reinforcement Learning, the Proximal Policy Optimization (PPO) algorithm, and custom reward shaping techniques.

# How it works?
AI sees the world as an array of 8 numbers (X/Y coordinates, velocities, tilt angle, and leg contact). By default, it struggles to land efficiently. I implemented a custom Python wrapper to intercept the physics engine and apply **Reward Shaping**.

Get to the ground Fix:- AI also used a loophole which would give it good amount of points if it landed rocket at ground so to get that massive points it started to ignore the final destination where rocket should land and it was just prioritzing landing the rocket on ground anywhere.

Burn Exploit Fix:- AI realised that buring more fule will result in decreased points so to get more reward it decided to freefall and then burst the engine when reaching ground and even after that burst it was very noticable that rocket was slamming into the ground. I had to fix it by reducing the fuel using penelty.

Hovering Loophole Fix:- AI is again trying to outsmart me and after fixing Burn Exploit Fix where i made using fuel a bit cheap and said to make vertical speed good for landing. Ai concluded that going down, landing with a perfect speed is hard so as fuel is cheap it will just float above the target. To fix this we introduce TIME TAX which will give AI and urgency to land which should fix this problem.
# Built With
**Python**:

**Curiosity**:




# Images

![Image1]()

![Image2]()

![Image3]()







# How to Contribute

The best way to contribute is simply by exploring the site! But if you want to help me improve it, I would love your feedback:

**Report Bugs:** Did you find a weird glitch, a broken link, or a layout issue? Let me know so I can fix it.

**Suggest Ideas:** Share your ideas for new features, hidden easter eggs, or cool retro web elements you would love to see added to the shrine.

**English Corrections:** I typed everything on this site myself, so there are probably a few spelling or grammar mistakes. If you spot any while reading, please point them out!

How to reach me:
You don't even need a GitHub account to contribute. Just use the Mail feature directly on the website to send me an email with your ideas, bug reports, or text corrections!

(If you are a developer, you are also welcome to open an Issue or a Pull Request right here on GitHub).



# Deployment

**Experience It Online**

Don't want to install anything or mess with code? No problem. You can explore the site directly in your browser!

Visit the Live Site: https://dixit106.hackclub.app/



**How to Run Locally**

Want to boot up this site on your own machine? It's super easy. You just need Python installed on your computer.

OPEN YOUR TERMINAL:-

1. Clone the repository:

```Bash
git clone git clone https://github.com/Dixit106/Dixit-s-DMN.git
```

2. Go into the folder:

```Bash
cd Dixit-s-DMN
```

3. Install Flask:
(Note: It is highly recommended to create a virtual environment first, but you can also just install it directly!)

```Bash
pip install flask
```

4. Start the server:

```Bash
python app.py
```
(Note: If your main python file is named something else, run that instead!)

5. Open your browser:
Go to http://127.0.0.1:5000 and start your journey!

