## Chapter 5

### Monte Carlo Methods

SUNJIANING Mar. 1st    Schulich

- Here we do not assume complete knowledge of environment. Monte Carlo methods require only `experience` — sample sequences of states, actions, and rewards from actual or simulated interaction with an environment.
- Although a model is required, the model need only generate sample transitions, not the complete probability distributions pf all possible transitions that is required for DP.
- Monte Carlo Methods are ways of solving the reinforcement learning problem based on averaging sample returns.

#### 5.1 Monte Carlo Prediction

An obvious way to estimate it from experience is simply to average the returns observed after visits to the state. As more returns are observed, the average should converge to the expected value. 

`first-visit MC: `estimates $v_\pi(s)$ as the average of the returns following first visits to , whereas `every-visit MC` method averages the returns following all visits to s.

![](https://ws4.sinaimg.cn/large/006tNc79gy1foy5g905z7j31e20gmtcs.jpg)

The DP backup diagram shows all possible transitions, the MC diagram shows only those sampled on the one episode. Whereas the DP diagram includes only one-step transitions, the MC diagram goes all the way to the end of the episode.

<img src="https://ws1.sinaimg.cn/large/006tNc79gy1fox0rkacfij30cw0bmt9y.jpg" width="250px"/>

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1foy5qtgf50j30440h4aa6.jpg" width="100px"/>

Estimates of MC methods for each state are **independent** and **do not bootstrap**.

#### 5.2 Monte Carlo Estimation of Action Values

If a model is not available, then it is particularly useful to estimate action values (the values of state–action pairs) rather than state values. 


Our primary goals for MC methods is to estimate $q_*$

**exploring starts: **In oreder to `maintain exploration`, we need to specify that the episodes start in a state-action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state-axtion pairs will be visited an infinite number of times in the limit of an infinite number of episodes.

#### 5.3 Monte Carlo Control

Control: to approximate optimal policies

To begin, let us consider a Monte Carlo version of classical policy iteration. In this method, we perform alternating complete steps of policy evaluation and policy improvement, beginning with an arbitrary policy $\pi_0$ and ending with the optimal policy and optimal action-value function:

![](https://ws3.sinaimg.cn/large/006tNc79gy1foy6nhab4pj30sa02ojrg.jpg)			
Policy evaluation is done exactly as described in the preceding section. Policy improvement is done by making the policy greedy with respect to the current value function./      

$$\pi(s)\doteq \arg\max\limits_aq(s,a)$$		
Policy improvement then can be done by constructing each $\pi_{k+1}$ as the greedy policy with respect to $q_{\pi_k}$. The policy improvement theorem in **Section 4.2** then applies to $\pi_k$	and $\pi_{k+1}$ becase for all $s\in S$:

$$q_{\pi_k}(s,\pi_{k+1}(s))\geq v_{\pi_k}(s)$$

As we discussed in the previous chapter, the theorem assures us that each $\pi_{k+1}$ is uniformly better than $\pi_{k}$, or just as good as $\pi_{k}$, in which case they are both optimal policies. This in turn assures us that theoverall process converges to the optimal policy and optimal value function. In this way Monte Carlomethods can be used to find optimal policies given only sample episodes and no other knowledge of theenvironment’s dynamics.

![](https://ws3.sinaimg.cn/large/006tNc79gy1foy6ysduqzj31fq0lajxq.jpg)

#### 5.4 Monte Carlo Control without Exploring Starts

`on-policy method:  ` attempt to evaluate or improve the policy that is used to make decision, whereas `off-policy methods` evaluate or improve a policy different from that used to generate the data

The $\varepsilon$-greedy policies are examples of $\varepsilon$-soft policies defined as policies for which $\pi(a|s)\geq \frac{\varepsilon}{|A(s)|}$ for all states and actions, for some $\varepsilon > 0$

![](https://ws1.sinaimg.cn/large/006tNc79gy1fp06i8qnjgj31ce0m47as.jpg)

Now we only achieve the best policy among the $\varepsilon$-soft policies, but on the other hand, we have eliminated the assumption of exploring starts.

#### 5.5 Off-policy Prediction via Importance Sampling

An more straightforward approach is to use two policies, one that is learned about and that becomes the optimal policy, and one that is more exploratory and is used to generate behavior. The policy being learned about is called the `target policy`, and the policy used to generate behavior is called the `behavior policy`. In this case we say that learning is from data 'off' the target policy and the overall process is termed `off-policy learning`.

We begin the study of off-policy methods by considering the *prediction* problem, in which both target and behavior policies are fixed. That is, suppose we wish to estimate $v_\pi$ or $q_\pi$, but all we have are episodes following another policy b, where $b\neq \pi$. In this case, $\pi$ is the target policy, b is the behavior policy, and both policies are cosidered fixed and given.

In oreder to use episodes from b to estimate values for $\pi$, we require that every action taken under $\pi$ is also taken at least occasionally, under b.

That is, we require that $\pi(a|s)>0$ impiles $b(a|s)>0$. This is called the assumption of *coverage*.

`importance sampling: ` **a general technique for estimating expected values under one distribution given samples from another.**

We apply importance sampling to off-policy learning by weighting returns ccording to the relative probability of their trajectories occurring under the target and behavior policies, called the `importance-sampling ratio`. Given a starting state $S_t$, the probability of the subsequent state-action trajectory, $A_t,S_{t+1},A_{t+1},…,S_T$, occuring under any policy $\pi$ is

![](https://ws4.sinaimg.cn/large/006tNc79gy1fp09k37xysj30z4082t9j.jpg)

where p here is the state-transition probability function. Thus, the relative probability of the trajectory under the target and behavior policies is 

![](https://ws2.sinaimg.cn/large/006tNc79gy1fp09l5gb6dj30u20463z2.jpg)

This is called the **importance sampling ratio**.In importance sampling, each return is weighted by the relative probability of the trajectory under the two policies.

All importance sampling ratios have expected value 1:

![](https://ws2.sinaimg.cn/large/006tNc79gy1fp0aekbt8yj30ps03e3yv.jpg)

Now we are ready to give a Monte Carlo algorithm that uses a batch of observed episodes following policy b to estimate $v_{\pi}(s)$.

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp09mnjzi5j314e0qqte7.jpg)

The ordinary importance-sampling estimator is unbiased whereas the weighted importance-sampling estimator is biased. On the other hand, the variance of the ordinary importance-sampling estimator is in general unbounded because the variance of the ratios can be unbounded, whereas in the weighted estimator the largest weight on any single return is ONE.

#### 5.6 Imcremental Implementation

The box on contains a complete episode-by-episode incremental algorithm for Monte Carlo policy evaluation. The algorithm is nominally for the off-policy case, using weighted importance sampling, but applies as well to the on-policy case just by choosing the target and behavior policies as the same (in which case (π = b), W is always 1). 

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp09yfo674j31gw0qwn1y.jpg)

*question: why here W = 0 then we should exit? and why we update W by multiplying with the importance sampling ratio? what's the meaning of that?*



#### 5.7 Off-policy Monte Carlo Control

In on-policy methods, they estimate the value of a policy while using it for control. In off-policy methods these two functions are separated. The policy used to generate behavior, called the behavior policy, may in fact be unrelated to the policy that is evaluated and improved, called the target policy. An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions.


![](https://ws1.sinaimg.cn/large/006tNc79gy1fp0ajsv867j31hc0rgdl6.jpg)

**Target policy is greedy and deterministic**

**Behavior policy is soft, typicaly $\varepsilon$-greedy**

question: why we need to exit the For loop when $A_t\neq \pi(S_t)$*



#### 5.8 Discounting-aware Importance Sampling

#### 5.9 Per-reward Importance Sampling























