## Introduction

SUNJIANING Feb 27th at Schulich 

**Part One - Tabular Solution Methods**: the state and action spaces are small enough for the approximate value functions to be represented as arrays, or tables.

| Chapter 2                                               | solution methods for bandit problems in which there is only a single state |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| **Chapter 3**                                           | **general problem formulation: finite Markov decision process along with Bellman equations and value functions** |
| **Chapter 4, 5, 6                                    ** | **three fundamental classes of methods for sovling finite MDP: dynamic programming, Monte Carlo methods, temporal-difference learning** |
| **Chapter 7     **                                      | **the strengths of MC methods combined with TD methods via eligibility traces** |
| **Chapter 8**                                           | **TD combines with model learning and planning methods**     |

## Chapter 2

### Multi-armed Bandits

Difference between RL and ML: RL uses training information that *evaluates* the actions taken rather than *instructs* by giving correct actions. Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible; Purely instructive feedback indicates the correct action to take, independently of the action actually taken.

#### 2.1 A k-armed Bandit Problem

`k-armed bandit problem:` You are faced repeatedly with a choice among `k` different options, or actions. After each choice you receive a numerial reward chosen from a stationary probability distribution that depends on the action you selected. You objective is to maximize the expected total reward over some time period, like over 1000 action selctions, or `time steps`.

$A_t$ — action selected on step t

$R_t$ — corresponding reward

$q_*(a)$ — expected reward given that a is selected

$$q_*(a)\doteq \mathbb E[R_t | A_t = a]$$

_question: what's the difference between A_t and q*_ 

We do not have the action values with certainty (otherwise it would be trivial to the solve the problem). We denote the estimated value of action a at time step t as $Q_t(a)$. We would like it to be close to $q_*(a)$.

If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We call these the `greedy actions`. When you select one of these actions, that is `exploitation`. If instead you select one of the nongreedy actions, that is `exploration`. Exploitation is the right thing to do to maximize the expected reward on the one stpe, but exploration may produce the greater total reward in the long term.

If you have many time steps ahead on which to make action selections, then it may be better to explore the nongreedy actions and discover which of them are better than the greedy action.

#### 2.2 Action-value Methods

True value of an action is the **mean reward** when that action is selected.

Hence, one way to estimate this is by averaging the rewards actually received:

$$Q_t(a)\doteq \frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a take prior to t}}=\frac{\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}} $$

where $\mathbb1_{predicate}$ denotes the random variable that is 1 if `predicate` is true and 0 if it is not. 

`sample-average: `If the denominator is zero, then we instead define $Q_t(a)$ as some default value, such as 0. As the denominator goes to infinity, by the law of large numbers, $Q_t(a)$ converges to $q_*(a)$. 

Then for greedy action selection: $A_t \doteq argmax_a Q_t(a)$

**$\varepsilon-greedy$**: behave greedily most of the time, but every once in a while with small probability $\varepsilon$, instead select randomly from among all the actions with equal probability, independently of the action-value estimates. An advantage of tat is in the limit as the number of steps increases, every action wil bee sampled an infinite number of times, thus ensuring that all the Q_t converge to q*.

#### 2.3 The 10-armed Testbed

This was a set a 2000 randomly generated k-armed bandit problems with k=10. For each bandit problem, such as the one shown in Figure, the action values,$q_*(a), a=1,…,10$, were selected according to a Gaussian distribution with mean 0 and variance 1. Then when a learning method applied to that problem elected action $A_t$ at time step t, the actual reward, $R_t$, was selected from a Gaussian distribution with mean $q_*(A_t)$ and variance 1. 

![](https://ws2.sinaimg.cn/large/006tNc79gy1fovx9njwi6j314q0ryn12.jpg)

The advantage of ε-greedy over greedy methods depends on the task. For example, suppose the
reward variance had been larger, say 10 instead of 1. With noisier rewards it takes more exploration to
find the optimal action, and ε-greedy methods should fare even better relative to the greedy method.
On the other hand, if the reward variances were zero, then the greedy method would know the true value of each action after trying it once. In this case the greedy method might actually perform best
because it would soon find the optimal action and then never explore. 

*Two factors in terms of ε-greedy and greedy method: stationary/nonstantionary, variance of reward*

#### 2.4 Incremental Implementation

The action-value methods we have discussed so far all estimate action values as sample averages of observed rewards. We want to find a more computationally efficient manner with constant memory and constant per-time-step computation.

It is easy to devise incremental formulas for updating averages with samll, constant computation required to process each new reward. 

Given $Q_n$ and the nth reward $R_n$, the new average of all nth rewards can be computed by:

 $Q_{n+1} = \frac{1}{n}\sum_{i=1}^nR_i = Q_n+\frac{1}{n}[R_n-Q_n]  (Q_2=R_1)$

The implementation requires memory only for $Q_n$ and n.			

![](https://ws4.sinaimg.cn/large/006tNc79gy1fovxt92as2j31b40eytcc.jpg)

The general form for update rule is *NewEstimate <— OldEstimate + StepSize [Target - OldEstimate]*

The expression [Target - OldEstimate] is an `error` in the estimate. It is reduced by taking a step toward the Target. The target is presumed to indicate a desirable direction in which to move. In the case above the target is the nth reward.

#### 2.5 Tracking a Nonstationary Problem

Stationary problem: the reward probabilities do not change over time

For nonstationary problem it makes sense to give more weight to recent rewards than to long-past reawrds.

weighted average: $$Q_{n+1} = (1-\alpha)^nQ_q + \sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i$$

the sum of the weights is $$(1-\alpha)^nQ_q + \sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i=1$$

`exponential recency-weighted average: `The weight given to the reward $R_i$, $\alpha(1-\alpha)^{n-i}$ depends on how many rewards ago, n-i, it was observed. As 1- $\alpha$ is less than 1, thus the weight given to $R_i$ decreases as the number of intervening rewards decreases. The weight decays exponentially according to the exponent on 1-$\alpha$.

The choice $\alpha_n(a)=\frac{1}{n}$ results in the sample-average method, which is guaranteed to converge to the true action values by the law of large numbers. But the convergence is not guaranteed for all choices of the sequence ${\alpha_n(a)}$. A well-known result in stochastic approximation theory gives us the conditions required to assure convergence with probability 1:

$$\sum_{n=1}^\infty \alpha_n(a)=\infty$$ and $$\sum_{n=1}^\infty \alpha_n^2(a)<\infty$$

#### 2.6 Optimistic Initial values

All the methods we have discussed so far are dependent to some extent on the initial action-value estimates, $Q_1(a)$. In the language of statistics, these methods are `biased` by their initial estimates.

Initial action values can be used as a simple way to encourage exploration. Suppose that instead setting the initial action values to zero, as we did in the 10-armed testbed, we set them all to +5.  Hence the initial estimate is thus wildly optimistic. But this optimism encourages action-value methods to explore. Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions, being "disappointed" with the rewards it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amout of exploration even if greedy actions are selected all the time.

Cons: we regard it as a simple trick that can be quite effective on stationary problems, but it is far from being a generally useful approach to encouraging exploration. It is not well suited to nonstationary problems because its drive for exploration is inherently temporary. If the task changes, creating a renewed need for exploration, this method cannot help.

#### 2.7 Upper-Confidence-Bound Action Selection

*Encourage exploration with preference rather than just arbitrary choose an nongreedy action*.

ε-greedy action selection forces the non-greedy actions to be tried, but indiscriminately, with no preference for those that are nearly greedy or particularly uncertain. It would be better to select among the non-greedy actions according to their potential for actually being optimal,taking into account both how close their estimates are to being maximal and the uncertainties in those estimates.

Formula: $$A_t\doteq argmax_a[Q_t(a)+c\sqrt \frac{lnt}{N_t(a)}]$$

t — time

N_t(a) — the number of times that action a has been selected prior to time t

c > 0 — the degree of exploration

The idea of this upper confidence bound action selection is that the square-root term is a measure of the `uncertainty` or `variance` in the estimate of a's value. All action will eventually be selected, but actions with lower value estimates, or that have already been selected frequently will be selected with decreasing frequency over time.

#### 2.8 Gradient Bandit Algorithms

*question: it's said that bandit problem only has one single state, what is that??*

*answer: there is only one vending machine with multi arms, so with actions it won't change to other state or other machines.*

We learn a numerial `preference` for each action a,  denote by $H_t(a)$, the larger the preference, the more often that action is taken.

The relative preference of ne action over another is important. Action probabilities are determined by a `soft-max distribution` .

$$Pr\{{A_t=a}\}\doteq\frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}\doteq \pi_t(a)$$

A natural learning algorithm for this setting based on the idea of stochastic gradient ascent. On each step, after selecting action $A_t$ and receiving the reward $R_t$, preferences are updated by:

$$H_{t+1}(A_t)\doteq H_{t+1}(A_t)+\alpha(R_t-\overline{R_t})(1-\pi_t(A_t))$$, and

$$H_{t+1}(a)\doteq H_{t+1}(a)+\alpha(R_t-\overline{R_t})\pi_t(a)$$, for all $a\neq A_t$

The $\overline{R_t}\in \mathbb{R}$ term serves as a `baseline` with which the reward is compared. **If the reward is higher than the baseline, then the probability of taking $A_t$ in the future is increased, and if the reward is below baseline, then probability is decreased.** The non-selected actions move in the opposite direction.

$$\overline{R_{t}}\doteq\frac{1}{t}\sum_{i=1}^tR_i$$

#### 2.9 Associative Search (Contextual Bandits)

`nonassociative task:` tasks in which there is no need to asociate different actions with different situations.

`policy`: a mapping from situations to the actions that are best in those situations.

`Associative task: ` suppose there are several different k-armed bandit tasks, and that on each step you confront one of these chosen at random. Thus the bandit task changes randomly from step to step. This would appear to you as a single, nonstationary k-armed bandit task whose trun action values change randomly from step to step. We can deal with this problem if we have **distinctive clue.** For instance, if read, select arm 1; if green, select arm 2.

This is an example of an `associative search` task, so called because it involves both trial-and-error learning to `search` for the best actions, and `association` of these actions with the situations in which they are best.

#### 2.10 Summary

Above are several simple ways of **balancing exploration and exploitation**. The $\varepsilon$-greedy methods choose randomly a small fraction of the time, whereas UCB methods choose deterministically but achieve exploration by subtly favoring at each step the actions that have so far received fewer samples; Gradient bandit algorithms estimate not action values, but action preferences, and favor the more preferred actions in a graded, probabliisic manner using a soft-max distribution; The simple expedient of initializing estimates optimistically causes even greedy methods to explore significantly.

`Bayesian` methods assume a known initial distribution over the action values and then update the distribution exactly after each step. In general, the update computations can be very complex, but for certain special distributions (called `conjugate priors`) they are easy. One possibility is to then select actions at each step according to their posterior probability of being the best action. This method called `posterior sampling` or `Thompson sampling`. 



![](https://ws3.sinaimg.cn/large/006tNc79gy1fowl6fe1hdj319e0lun1o.jpg)



​	

​	




​			
​		
​	































