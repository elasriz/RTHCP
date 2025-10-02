# RT-HCP: Dealing with Inference Delays and Sample Efficiency to Learn Directly on Robotic Platforms [Paper](https://arxiv.org/abs/2509.06714)

Authors: Zakariae El Asri, Ibrahim Laiche, Clément Rambour, Olivier Sigaud, Nicolas Thome  
Affiliation: Sorbonne Université, CNRS, ISIR, Paris, France  

## 🧠 Motivation

Reinforcement Learning (RL) has shown great success in simulated environments, but faces two major bottlenecks in real-world robotic applications:

- **Sample Inefficiency**: Model-free RL requires a large amount of interactions with the environment, which is impractical or costly on physical robots.
- **Inference Delays**: In model-based RL, computing the next action using forward models or planning methods often takes too long to maintain real-time control on embedded hardware.

## ⚙️ Key Contributions

- **D-step MPC Framework**: A novel method to handle **inference delays** in model-based reinforcement learning by computing and queuing several actions in advance.
- **RT-HCP**: A **real-time hybrid control** policy that combines:
  - model-based planning,
  - model-free actor-critic agent for fast feedback,
  - and prior dynamics models,


## 📐 Method Overview

Our pipeline introduces:
- A **Delay Wrapper** that explicitly models and handles inference delays using action and state buffers.
- An **augmented state** formulation that incorporates both current and hidden past states for delay-aware control.
- A **D-step MPC strategy** that precomputes sequences of actions to bridge inference gaps.

<p align="center">
  <img src="media/RTHCP_method.gif" alt="RT-HCP Diagram" width="600"/>
</p>

## 📊 Experimental Results

RT-HCP was evaluated on a real Furuta pendulum system, and compared to several baselines:

| Method     | Performance | Real-time capability | Sample efficiency| 
|------------|-------------|----------------------|------------------|
| RT-HCP     | 🟢 High     | ✅ Yes                | 🟢 High          |
| RT-TDMPC   | 🟠 Moderate | ✅ Yes                | 🟠 Moderate      |
| RT-PETS    | ⚠️ Low      | ⚠️ No                 | 🟠 Moderate      |
| TD3        | 🟠 Moderate | ✅ Yes                | ⚠️ Very Low      |

RT-HCP enables continuous control and robust policy learning, even under strict real-time constraints.

## 🚀 How to Use

### 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rt-hcp.git
   cd rt-hcp

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> ⚠️ Note:
- This code is designed to interface with the Quanser Qube-Servo 2 platform. Ensure that the Quanser HIL API and drivers are correctly installed.
- ✅ The system has been tested on Windows only. While parts may work on other platforms, real hardware interaction depends on Quanser libraries available on Windows.

### ▶️ Launch Training

To start a training session on the real Furuta pendulum:
```bash
python train_rthcp.py
```

You can adjust parameters such as delay, seed, episode length, etc., directly in the script.

## 📽 Project Video

A short presentation is available: [Watch on YouTube](https://youtu.be/Janb7beQVwk) 

---

## 🤖 Acknowledgments

This work is supported by the European Commission’s Horizon Europe Framework Programme under grant agreement No. 101070381 (PILLAR-robots project).
