## Chapter 10

### On-policy Control with Approximation

SUNJIANING   Mar. 4th 	HSSL

- In this chapter we return to the control problem, now with parametric approximation of the action-value function $\hat q(s,a,\mathbb{w})\approx q_*(s,a)$, where $\mathbb{w}\in \mathbf{R}^d$ is a finite-dimensional weight vector.
- Suprisingly, once we have genuine function approximation we have to give up discounting and switch to a new "average-reward" formulation of the control problem, with new "differential" value functions.

#### 10.1 Episodic Semi-gradient Control

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fp1dszowp0j30pk0a83zp.jpg" width="400px" />

Always learn the action-value function of the current policy

Always act near-greedily wrt the current action-value estimates

The learning rule is the same as in Chapter 9:

$$\mathbb{w}_{t+1}\doteq \mathbb{w}_{t+1}+\alpha\big[U_t-\hat q(S_t, A_t, \mathbb{w}_t)\big]\nabla \hat q(S_t,A_t,\mathbb{w}_t)$$

MC: $U_t=G_t$

Sarsa: $U_t = R_{t+1}+\gamma \hat q\big(S_{t+1}, A_{t+1}, \mathbb{w}_t\big)$

Expected Sarsa: $U_t=R_{t+1}+\gamma \sum\limits_a \pi(a|S_{t+1})\hat q(S_{t+1},a,\mathbb{w}_t)$

DP: $U_t = \sum\limits_{s',r}p(s',r|S_t,A_t)\big[r+\gamma \sum\limits_{a'}\pi(a'|s')\hat q(s',a',\mathbb{w}_t)\big]$

To form control methods, we need to couple such action-value prediction methods with techniques for policy improvement and action selection. Suitable techniques applicable to continuous actions, or to actions from large discrete sets, are a topic of ongoing research with as yet no clear resolution. (DDPG?)


On the other hand, if the action set is discrete and not too large, then we can use the techniques already developed in previous chapters. That is, for each possible action a available in the current state St, we can compute qˆ(St,a,wt) and then find the greedy action $A_t^*=\arg\max_a\hat q(S_t,a,\mathbb{w})t$. Policy improvement is then done (in the on-policy case treated in this chapter) by changing the estimation policy to a soft approximation of the greedy policy such as the ε-greedy policy. Actions are selected according to this same policy.

![](https://ws2.sinaimg.cn/large/006tKfTcgy1fp1e2iws2gj31g20ne797.jpg)			
​		

#### 10.2 n-step Semi-gradient Sarsa

The n-step return immediately generalizes from its tabular form to a function approximation form:

![](https://ws3.sinaimg.cn/large/006tKfTcgy1fp1ekxfbwij31cm06ita3.jpg)

​	

#### 10.3 Average Reward: A New Problem Setting for Continuing Tasks

This is no discounting - the agent cares just as much about delayed rewards as it does about immediate reward. 

In the average-reward setting, the **quality** of a policy $\pi$ is defined as the average rate of reward while following that policy, which we denote as $r(\pi)$:

<img src="https://ws1.sinaimg.cn/large/006tKfTcgy1fp1etytwqgj30oa0a4my9.jpg" width="500px" />

where the expectations are conditioned on the prior actions, $A_0,A_1,…,A_{t-1}$, being taken according to $\pi$, and $\mu_{\pi}$ is the steady-state distribution

<img src="https://ws1.sinaimg.cn/large/006tKfTcgy1fp1eyfuy9aj30h402674d.jpg" width="350px" />

In the average-reward setting, returns are defined in terms of differenes between rewards and the average reward:

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fp1f22rl13j30ue02ct8t.jpg)

This is known as the `differential return`, and the corresponding value functions are known as `differential value functions`.

![](https://ws2.sinaimg.cn/large/006tKfTcgy1fp1f31eu7mj310a0bidhi.jpg)

![](https://ws3.sinaimg.cn/large/006tKfTcgy1fp1f3v63fcj30wu04u3z4.jpg)

![](https://ws2.sinaimg.cn/large/006tKfTcgy1fp1f4dzg7pj31go0nyn1y.jpg)

#### 10.4 Deprecating the Discounted Setting

Discounting is futile in continuing control settings with function approximation.

- why? to be con'd...

#### Conclusions

- Control is straightforward in the on-policym episodic, linear case
- For the continuing case, we need the average-reward setting
  - which is a lot like just replacing $R_t$ with $R_t - r(\pi)$ everywhere
  - where $r(\pi)$ is the average reward per step, or its estimate
- We should probably never use discounting as a control objective
- Formal results exist for the linear, on-policy case
  - we get chattering near a good solution, not convergence

















