# Tutorials on Control Barier Functions

In this series of tutorials, we present a short introduction on Control Barrier Functions (CBF) for start-to-goal motion tasks. CBFs are continuous functions whose value on a point increases to infinity as the point approaches the boundary of the feasible region of an optimization problem. In general, CBFs:

- can be utilized to synthesize fast controllers that operate in real time.
- can serve as safety filters to legacy controllers.

The jupyter notebooks explain step-by-step the control synthesis process. If for any reason, the videos are not generated, please run the python version of the file.

For environment setup, please follow the instructions at the bottom of this document.

### CBF example: Linear system, static obstacles

[01-Tutorial-CBF-Static-Obstacles.ipynb](01-Tutorial-CBF-Static-Obstacles.ipynb) (code and explanation)

[01-Tutorial-CBF-Static-Obstacles.py](01-Tutorial-CBF-Static-Obstacles.py) (code)

[01A-Tutorial-CBF-Static-Obstacles.py](01A-Tutorial-CBF-Static-Obstacles.py) (more obstacles)

[01-Tutorial-CBF-Static-Obstacles-superellipse.py](01-Tutorial-CBF-Static-Obstacles-superellipse.py) (obstacles as superellipses for a continous approximation of rectangles. Continuity needed for CBF derivation)

### CBF example: Linear system, dynamic obstacles

[02-Tutorial-CBF-Dynamic-Obstacles.ipynb](02-Tutorial-CBF-Dynamic-Obstacles.ipynb) (code and explanation)

[02-Tutorial-CBF-Dynamic-Obstacles.py](02-Tutorial-CBF-Dynamic-Obstacles.py) (code)

[02A-Tutorial-CBF-Dynamic-Obstacles.py](02A-Tutorial-CBF-Dynamic-Obstacles.py) (more obstacles)

### Higher-order CBF example: Nonlinear bicycle system, static obstacle
[03-Tutorial-HOCBF-Static-Obstacles.ipynb](03-Tutorial-HOCBF-Static-Obstacles.ipynb) (code and explanation)

[03-Tutorial-HOCBF-Static-Obstacles.py](03-Tutorial-HOCBF-Static-Obstacles.py) (code)

## Environment setup

Clone the repository and install the dependencies. 

```git
git clone https://github.com/bardhh/control_barrier_function_kit.git
```

Make sure you are in the tutorial branch.
```git
git checkout tutorial
```

Install ffmepeg for saving animation. On Ubuntu, run:

```bash
sudo apt install ffmpeg
```

Install necesarry python packages.

```bash
pip install -r requirements.txt
```

## [Optional] VScode settings

Use VScode to follow the tutorials.

### VScode extentions
Useful extensions

1. Jupyter
2. Markdown All in One

## Reference Papers

### References
[01 - Safety Verification of Hybrid Systems Using Barrier Certificates](http://web.mit.edu/~jadbabai/www/papers/hscc04_2.pdf)  

[02 - Control Barrier Functions: Theory and Applications](https://arxiv.org/pdf/1903.11199.pdf)  

[03 - Control Barrier Functions for Systems with High Relative Degree](https://arxiv.org/pdf/1903.04706.pdf)

[04 - Risk-bounded control using stochastic barrier functions](https://www.bhoxha.com/papers/LCSS2020.pdf) (CPS group)

[05 - Safe Navigation in Human Occupied Environments Using Sampling and Control Barrier Functions](https://arxiv.org/pdf/2105.01204.pdf)  (CPS group)

[06 - Risk-Bounded Control with Kalman Filtering and Stochastic Barrier Functions]()  (CPS group)

### Applications
**Legged robots**

[07 - 3D Dynamic Walking on Stepping Stones with Control Barrier Functions](https://ece.umich.edu/faculty/grizzle/papers/3DSteppingStones_CDC2016.pdf)

[08 - Multi-Layered Safety for Legged Robots via Control Barrier Functions and Model Predictive Control](https://arxiv.org/pdf/2011.00032.pdf)

**AMS**

[09 - Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function](https://arxiv.org/pdf/2007.11718.pdf)

[10 - Safe teleoperation of dynamic uavs through control barrier functions](https://hybrid-robotics.berkeley.edu/publications/ICRA2018_Safe_Teleoperation.pdf)

**Multi-agent systems**

[11 - Safety Barrier Certificates for Collisions-Free Multirobot Systems](http://ames.caltech.edu/wang2017safety.pdf)

**COVID**

[12 - Safety-Critical Control of Compartmental Epidemiological Models with Measurement Delays](https://arxiv.org/pdf/2009.10262.pdf)


