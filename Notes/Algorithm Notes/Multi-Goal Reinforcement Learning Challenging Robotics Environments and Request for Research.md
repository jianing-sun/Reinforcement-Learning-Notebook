## [Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research](https://d4mucfpksywv.cloudfront.net/research-covers/ingredients-for-robotics-research/technical-report.pdf)

This paper from OpenAI is basiclly a technical report for their experiments in a newly introduced environment **Robotics** in gym which is extended from MuJoCo but mainly for robotics tasks, such as FetchReach, PickAndPlace, etc and also for hand imitation experiments.

- Fetch environments

  - The Fetch environments are based on the 7-DoF Fetch robotics arm, which has a two-fingered parallel gripper. 
  - **Goal**: 3-dimensional and describes the desired position of the object (or the end-effector for reaching);
  - **Reward** are sparse and binary: the agent obtains a reward of 0 if the object is at the target location (within a tolerance of 5cm) and -1 otherwise. Actions are 4-dimensional: 3 dimensions specify the desired gripper movement in Cartesian coordinates and the last dimension controls opening and closing of the gripper;
  - **Actions** are 4-dimensional: 3 dimensions specify the desired gripper movement in Cartesian coordinates and the lsat dimension controls opening and closing of the gripper; They **apply the same action in 2- subsequent simulator steps (with ∆t = 0.002 each) ** before returning control to the agent (f=25Hz).
  - **Observations** include the Cartesian position of the gripper, its linear velocity as well as the position and linear velocity of the robot's gripper. If an **object** is included, we also include the object's Cartesian position and rotation using Euler angles, its linear and angular velocities, as well as its position and linear velocities relateive to gripper. 

- Specific tasks

  ![](https://ws2.sinaimg.cn/large/006tKfTcgy1fpu1r1c4bsj315q0acq7f.jpg)

  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpu1rgaclvj314k0fan1d.jpg)

- Multi-goal environment interface

  All environments use goals that describe the desired outcome of a task. For example, in the
  FetchReach task, the desired target position is described by a 3-dimensional goal. While our
  environments are fully compatible with the OpenAI Gym API, we slightly extend upon it to support this new type of environment. All environments extend the newly introduced gym.GoalEnv.

  - The implementation is depend of **three environments in gym: MuJoCo_Env, Robot_Env, and Goal_Env**.

  - Goal-aware observation space: the observation space is of type `gym.spaces.Dict` space (not the same with dict), with at least following three key:

    1) **observation**: The actual observation of the environment, For example robot state and
    position of objects.

    2) **desired_goal**: The goal that the agent has to achieve. In case of FetchReach, this wouldbe the 3-dimensional target position.

    3) **achieved_goal**: The goal that the agent has currently achieved instead. In case ofFetchReach, this is the position of the robots end effector. Ideally, this would be thesame as desired_goal as quickly as possible.

  - (? TODO) **Exposed reward function:** Second, they expose the reward function in a way that allows for re-computing the reward with different goals. This is a necessary requirement for **HER-style** algorithmswhich substitute goals. A detailed example is available in Appendix A

- Experiments

  They implemented with DDPG algorithms with and without HER (Hintsight Experience Replay). I'm currently doing experiments with TRPO and going to combine HER with TRPO.

  So the following description of experiments currently base on their DDPG experiments and later on I will complement with my experiment results.

  ![](https://ws1.sinaimg.cn/large/006tKfTcgy1fpu3vuqkvlj31kw14ek6a.jpg)

  Figure 3 depicts the median test success rate for all four Fetch environments. FetchReach is clearly a very simple environment and can easily be solved by all four configurations. On the remaining environments, DDPG+HER clearly outperforms all other configurations. Interestingly, DDPG+HER performs best if the reward structure is sparse but is also able to successfully learn from dense rewards. For **vanilla DDPG **(what's this?), it is typically easier to learn from dense rewards with sparse rewards being more challenging.


  We evaluate the performance after each epoch by performing 10 deterministic test rollouts per MPI worker and then compute the test success rate by averaging across rollouts and MPI workers. Our implementation is available as part of OpenAI Baselines7 (Dhariwal et al., 2017). In all cases, we repeat an experiment with 5 different random seeds and report results by computing the median test success rate as well as the interquartile range.

  - A snippet of code about how to make Robotics env compatible with standard RL algorithms

```python
import gym
env = gym.make(’FetchReach−v0’) print(type(env. reset ()))
# prints "<class ’dict’>"
env = gym. wrappers . FlattenDictWrapper ( env, [’observation’, ’desired_goal’])
ob = env.reset()
print(type(ob), ob.shape)
# prints "<class ’numpy. ndarray’> (13 ,)"
```

- Hyperparameters

  For all environments, we train on a single machine with 19 CPU cores. Each core generates experience using two parallel rollouts and uses MPI for synchronization. For FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, and HandReach, we train for 50 epochs (one epoch consists of 19 · 2 · 50 = 1 900 full episodes), which amounts to a total of 4.75 · 106 timesteps.	

  ![](https://ws4.sinaimg.cn/large/006tKfTcgy1fpu3rq9076j31fc0eiq6r.jpg)