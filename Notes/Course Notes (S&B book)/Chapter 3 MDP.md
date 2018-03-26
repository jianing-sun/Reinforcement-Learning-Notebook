## Chapter 3 

### Finite Markov Decision Process

SUNJIANING Feb 28th McConnell 544

Compared with Bandit problems we estimated the value $q_*(a)$ of each action a, in MDPs we estimate the value $q_*(s, a)$ of each action a in each state s, or we estimate the value $v_*(s)$ of each state given optimal action selections.

#### 3.1 The Agent-Environment Interface

The learner and decision maker is called the `agent`. The thing it interacts with , comprising everyhing outside the agent, is called the `environment`. The MDP and agent together thereby give rise to a sequence or `trajectory` that begins like this:

$$S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,...$$

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fowpzgcb04j30wg0bmdhc.jpg" width="650px"/>

`Finite MDP: `the sets of states, actions, and rewards all have a finite number of elements. In this case the random variables $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action.

$$p(s',r|s,a)\doteq \text {Pr}\{{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a}\}$$

The probabilities given by the four-argument function p completely characterize the dynamics of a finte MDP.

`expected rewards:` $r(s,a)\doteq \mathbb E[R_t|S_{t-1}=s, A_{t-1}=a]=\sum_{r\in \mathcal{R}}r\sum_{s'\in \mathcal{S}}p(s',r|s,a)$ 

#### 3.2 Goals and Rewards

The agent's goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward but cumulative reward in the long run.

#### 3.3 Returns and Episodes

Formally, we seek to maximize the `expected return`, where the return, denoted $G_t$, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

**For episodic tasks** $$G_t\doteq R_{t+1}+R_{t+2}+R_{t+3}+….+R_T$$, where T is the final step

`Episode:` the agent-environment interaction breaks naturally into subsequences. The next episode begins independently of how the previous on ended. Tasks with episodes of this kind are called `episodic tasks`.

For `conitunious tasks`, then above-mentioned forumula $G_t$ would be unapplicable. Hence, in general, we use another definition of return:

**For continuous tasks **$G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+… = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$

where $\gamma$ is a parameter, $0\leq\gamma \leq1$ called the `discount rate`.

The discount rate determines the present value of futurn rewards: **reward received k time steps in the future is worth only $\gamma^{k-1}$ times what it would be worth if it were received immediately.** In general, acting to maximize immediate reward can reduce access to future rewards so that the return is reduced. As $\gamma$ approaches 1, the return objective takes future rewards into account more strongly; the agent becomes more farsighted.

*question: why we need discount rate?*

*my answer: discount rate makes future reward worth less than immediate reward as tomorrow you might die. That means the bigger the discount rate (can't bigger than 1 as we need it to converge), the more important the future is. Apprantly, if the gamma equals zero, then $G_t=R_t$ means return reward only depends on immediate reward.*

Returns at successive time steps are related to each other in a way that is important for the theory and algorithms of reinforcement learning:

$$G_t\doteq R_{t+1}+\gamma G_{t+1}$$

Note that although the return is a sum of an infinite number of terms, it is still finite if the reward is nonzero and constant if $\gamma < 1$. For example, if the reward is a constant +1, then the return is 

$$G_t=\sum_{k=0}^\infty \gamma^k=\frac{1}{1-\gamma}$$

#### 3.4 Unified Notation for Episodic and Continuing Tasks

We have defined the return as a sum over a finite number of terms in one case and as a sum over an infinite number of terms in the other. These can be **unified** by considering episode termination to be the entering of a special `absorbing state` that transitions only to itself and that generates only rewards of zero.

![](https://ws4.sinaimg.cn/large/006tNc79gy1fowzcxdk9lj30xw06yt9g.jpg)

$$G_t\doteq \sum_{k=t+1}^T \gamma^{k-t-1}R_k$$ including the possibility that $T=\infty$ or $\gamma =1$ but not both

*question: essential difference between the former formula?*

#### 3.5 Policies and Value Functions

`value functions:` state-action pairs, estimate how good it is for the agent to be in a given state. **A *policy* is a mapping from states to probabilities of selecting each possible action**.

**$v_\pi(s)$** — `state-value function`: it is the expected return when starting in s and following $\pi$ thereafter.

$$v_\pi (s)\doteq \mathbb E_{\pi}[G_t|S_t=s]=\mathbb E_\pi[\sum_{k=0}^\infty\gamma^kR_{t+k+1}|S_t=s]$$

Note that the value of the terminal state if any is always zero. 

Similarly, we define the value of taking action a in state s under a policy $\pi$ denoted $q_\pi(s,a)$, as the expected return starting from s, taking the action a, and thereafter following policy $\pi$

$$q_\pi (s)\doteq \mathbb E_{\pi}\big[G_t|S_t=s,A_t=a\big]=\mathbb E_\pi\big[\sum_{k=0}^\infty\gamma^kR_{t+k+1}|S_t=s,A_t=a\big]$$

We call the function $v_\pi$ the **state-value function for policy $\pi$**

​	      the function $q_\pi$ the **action-value function for policy $\pi$**

* question: what's the difference between these two functions?

A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that they *satisfy recursive relationships*. For any policy $\pi$ and any state s, the following consistency condition holds between the value of s and the value of its possible successor states:

**Bellman equation for $v_\pi$: **

$$ v_{\pi}(s)\doteq \mathbb{E}[G_t|S_t=s]=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')]$$, for all $s\in S$

Explanation of this formula: it is really a sum over all values of the three variables, a, s' and r. For each triple, we compute its probability, $\pi(a|s)p(s',r|s,a)$, weight the quantity in brackets by that probability, then sum over all possinbilities to get an expected value.

The Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state mush equal the (discounted) value of the expected next state, plus the reward expected along the way.

<img src="https://ws1.sinaimg.cn/large/006tNc79gy1fox0rkacfij30cw0bmt9y.jpg" width="250px"/>

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fox0yfdw68j307g08et93.jpg" width="200px"/>

We call diagrams like that above `backup diagrams` because they diagram relationships that form the basis of the update or backup operations that are at the heart of reinforcement learning methods. These operations transfer value information back to a state (or a state-action pair) from its successor states (or state-action pairs).

#### 3.6 Optimal Policies and Optimal Value Functions

`optimal policy:` better than or equal to all other policies, denote by $\pi_*$. They share the same state-value function and action-value function, called the `optimal state-value function`, denote by $v_*$ and $q_*$

$$v_*(s)\doteq \max\limits_{\pi}v_{\pi}(s)$$

$$q_*(s,a)\doteq \max\limits_{\pi}q_\pi (s,a)$$

These are the same as the backup diagramas for $v_\pi$ and $q_\pi$ presented earilier except that arcs have been added at the agent's choice points to represent that the maximum over that choice is taken rather than the expected value given some policy.



<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fox1heuen3j30pg09udh9.jpg" width="550px" />

**Bellman optimality equation:** express the fact that the value of a state under an optimal policy mush equal the expected return ofr the best action from that state.

$$v_*(s)=\max\limits_{a\in A(s)}q_{\pi_*}(s,a)=\max\limits_{a}\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_*(s')\big]$$

$$q_*(s,a)=\mathbb{E}\big[R_{t+1}+\gamma \max\limits_{a'}q_*(S_{t+1},a')\big|S-t=s, A_t=a\big]=\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma \max\limits_{a'}q_*(s',a')\big]$$ 

#### 3.7 Optimality and Approximation

Two defects: high computation cose and cost of memory

The memory available is an important constraint. A large amount of memory is often required to build up approximations of value functions, policies and models. In tasks with small, finite state sets, it is possible to form these approximations using arrays or tables with one entry for each state/state-action pair. This we call the `tabular` case.

#### 3.8 Summary

The undiscounted formulation is appropriate for episodic tasks, in which the agent-environment interaction breaks naturally into episodes; the discounted formulation is appropriate for continuing tasks.









































