# Uncertainty Sequence Modeling for Safe and Effective Autonomous Driving

## abstract
Deep reinforcement learning(RL) is an end-to-end data-driven control method that is widely used in the autonomous driving domain. However, conventional RL approaches have difficulties in applying it to autonomous driving tasks due to problems such as inefficiency, instability, and uncertainty. These issues play an important role in the autonomous driving domain. Although recent studies have attempted to solve these problems, they are computationally expensive and rely on special assumptions. In this paper, we propose a new algorithm MCDT that considers inefficiency, instability, and uncertainty by introducing a method called uncertainty sequence modeling to autonomous driving domain. The sequence modeling method, which views reinforcement learning as a decision making generation problem to obtain high rewards, avoids the disadvantages of exiting studies and guarantees efficiency, stability and also considers safety by integrating uncertainty estimation techniques. The proposed method was tested in the OpenAI Gym CarRacing environment, and the experimental results show that the MCDT algorithm provides efficient, stable and safe performance compared to the existing reinforcement learning method.

## Monte-Carlo Decision Transformer Scheme

![png](https://github.com/LoteeYoon/Uncertainty_Sequece_Modeling/blob/main/MCDT_scheme.PNG?raw=true)
