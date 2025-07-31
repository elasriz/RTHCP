---
title: RT-HCP: Dealing with Inference Delays and Sample Efficiency to Learn Directly on Robotic Platforms
---

# 🧠 Real-Time Hybrid Control with Physics

**Presented at [IROS 2025]**  
[[📄 Paper](https://drive.google.com/file/d/1o7KSa_mcdQYd44YF6MeNFWr5cQxrvxb7/view?usp=sharing)] · [[💻 Code](https://github.com/elasriz/RTHCP/)] · [[🎥 Video](https://youtu.be/Janb7beQVwk)]

---

##  🧠 Motivation


Reinforcement Learning (RL) has shown great success in simulated environments, but faces two major bottlenecks in real-world robotic applications:

- **Sample Inefficiency**: Model-free RL requires a large amount of interactions with the environment, which is impractical or costly on physical robots.
- **Inference Delays**: In model-based RL, computing the next action using forward models or planning methods often takes too long to maintain real-time control on embedded hardware.

---

## ⚙️ Key Contributions

- **D-step MPC Framework**: A novel method to handle **inference delays** in model-based reinforcement learning by computing and queuing several actions in advance.
- **RT-HCP**: A **real-time hybrid control** policy that combines:
  - model-based planning,
  - model-free actor-critic agent for fast feedback,
  - and prior dynamics models,

---

## 📐 Method Overview

Our pipeline introduces:
- A **Delay Wrapper** that explicitly models and handles inference delays using action and state buffers.
- An **augmented state** formulation that incorporates both current and hidden past states for delay-aware control.
- A **D-step MPC strategy** that precomputes sequences of actions to bridge inference gaps.

<p align="center">
  <img src="media/RTHCP_method.gif" alt="RT-HCP Diagram" width="600"/>
</p>

---

## 📊 Experimental Results

RT-HCP was evaluated on a real Furuta pendulum system, and compared to several baselines:

| Method     | Performance | Real-time capability | Sample efficiency| 
|------------|-------------|----------------------|------------------|
| RT-HCP     | 🟢 High     | ✅ Yes                | 🟢 High          |
| RT-TDMPC   | 🟠 Moderate | ✅ Yes                | 🟠 Moderate      |
| RT-PETS    | ⚠️ Low      | ⚠️ No                 | 🟠 Moderate      |
| TD3        | 🟠 Moderate | ✅ Yes                | ⚠️ Very Low      |

RT-HCP enables continuous control and robust policy learning, even under strict real-time constraints.

🎥 **Watch the policy behaviors:** [Link to video](#)

---

## 🔗 Resources

- 📄 **Paper**: [PDF](#)
- 💻 **Code**: [GitHub Repository](#)
- 📍 **BibTeX**:
```bibtex
@inproceedings{...}
