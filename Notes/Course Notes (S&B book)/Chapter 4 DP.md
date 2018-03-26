## Chapter 4

### Dynamic Programming

SUNJIANING Mar. 1st  Schulich

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies **given a perfect model** of the environment as a MDP.

Cons: assumption of a perfect model and great computational expense.

**key idea: **the use of value functions to organize and structure the search for good policies

`Bellman Optimality Equation:`

$v_*(s)=\max\limits_a\mathbb{E}\big[R_{t+1}+\gamma v_*(S_{t+1})\big|S_t=s,A_t=a\big]=\max\limits_a\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_*(s')\big]$

or

$$q_*(s,a)=\mathbb{E}\big[R_{t+1}+\gamma \max\limits_{a'}(S_{t+1},a')\big|S_t=s,A_t=a\big]=\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma \max\limits_{a'}q_*(s',a')\big]$$

**We can easily obtain optimal policies once we have found the optimal value functions, $v_*$ or $q_*$, which satisfy the Bellman Optimallity Equations.**

#### 4.1 Policy Evaluation (Predication)

`policy evaluation: `compute the state-value function $v_{\pi}$

Consider a sequence of `approximate value functions` $v_0,v_1,v_2,…$, each mapping $S^+$ to $\mathbb {R}$. The initial approxmation $v_0$, is chosen arbitrarily and each successive approximation is obtained by using the Bellman equation for $v_\pi$ as an `update rule`:

$$v_{k+1}(s)\doteq \mathbb{E_\pi}\big[R_{t+1}+\gamma v_k(S_{t+1})\big|S_t=s\big]=\sum\limits_{a}\pi(a|s)\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_k(s')\big]$$

`iterative policy evaluation: `the sequence ${v_k}$ can be shown in general to converge to $v_\pi$ as $k\rightarrow \infty$ under the same conditions that guarantee the existence of $v_{\pi}$. 

![](https://ws3.sinaimg.cn/large/006tNc79gy1foxae4gz7hj316u0eswgr.jpg)

### 4.2 Policy Improvement

**key criterion:** if $q_\pi(s,a)\doteq\mathbb{E}\big[R_{t+1}+\gamma v_\pi(S_{t+1})\big|S_t=s,A_t=a\big]$ is greater than or less than $v_\pi(s)$—— that is, if it is better to select a once in s and thereafter follow $\pi$ than it would be to follow $\pi$ all the time.

`policy improvement theorem:`

$$q_{\pi}(s,\pi'(s))\geq v_{\pi}(s)$$

Then the policy $\pi'$ mush be as good as, or better than, $\pi$. That is, it must obtain greater or equal expected return from all states $s\in S$:

$$v_{\pi'}(s)\geq v_{\pi}(s)$$

Hence, given a policy and its value function, we can easily evaluate a change in the policy at a single state to a particular action. It is a natural extension to consider changes at all states and to all posible actions, selecting at each state the action that appears best according to $q_\pi(s,a)$. In ohter words, to consider the new *greedy policy*, $\pi'$, given by

$$\pi'(s)\doteq \arg\max\limits_{a}q_\pi(s,a)=\arg\max\limits_{a}\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_{\pi}(s')\big]$$

`policy improvement:` process of making a new policy that improves on an original policy, by making it greedy w.r.t the value function of the original policy

Suppose the new greedy policy, $\pi'$ is as good as but not better than the old policy $\pi$. Then $v_\pi=v_{\pi'}$, from last equation we know that 

$$v_{\pi'}(s)=max\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_{\pi'}(s')\big]$$

This this the SAME as the Bellman Optimality Equation, and therefore, $v_{\pi'}(s)$ must be $v_{*}$, and both $\pi$ and $\pi'$ must be optimal policies. **Policy improvement thus must give us a strictly bettwe policy except when the original policy is already optimal.**

#### 4.3 Policy Iteration

Once a policy $\pi$ has been improved using $v_{\pi}$ to yield a better policy $\pi'$, we can then compute $v_{\pi'}$ and improve it again to yield an even better $\pi''$. We can thus obtain a sequence of monotonically improving policies and value functions:

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1foxza02k0ij30pa02mjry.jpg" width="700px" />

This way of finding an optimal policy is called `policy iteration`.

![](https://ws3.sinaimg.cn/large/006tNc79gy1foxzdsqm9sj318q0qy7am.jpg)

#### 4.4 Value Iteration

One drawback to policy iteration is that each of its iterations involves policy evaluation, which may it itself be a protracted iterative computation requiring multiple sweeps through the state set.

`value iteration: `In fact, the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state).

$$v_{k+1}(s)\doteq \max\limits_{a}\mathbb{E}\sum\limits_{s',r}p(s',r|s,a)\big[r+\gamma v_k(s')\big]$$

Value iteration update is identical to the policy evaluation update except that it requires the maximum to be taken over all actions. In value iteration, only a simple iteration of policy evaluation is performed in between each policy improvement.

![](https://ws1.sinaimg.cn/large/006tNc79gy1foy3cdasydj318e0hsq5z.jpg)

#### 4.5 Asynchronous Dynamic Programming

A major drawback to the DP methods that we have discussed so far is that they involve operations over the entire state set of the MDP, that is, they require sweeps of the state set. If the state set is very large, then even a single sweep can be prohibitively expensive.

`Asynchronous DP` does not use sweeps. Instead it works like this:

```markdown
Repeat util convergence criterion is met:
	- Pick a state at random and apply the appropriate backup
```

Still need lots of computation, but does not get locked into hopelessly long sweeps.

#### 4.6 Generalized Policy Iteration

![](https://ws1.sinaimg.cn/large/006tNc79gy1foy48ybcqbj31cw0o4tdh.jpg)

#### 4.8 Summary

- *Policy evaluation* refers to the iterative computation of the value functions for a given policy. *Policy improvement* refers to the computation of an improved policy given the value function for that policy. Puttin ghtese two computations together, we obtain *policy iteration and *value iteration*, thetwo most popular DP methods.
- Insight into DP methods and, in fact, into almost all reinforcement learning methods, can be gained by viewing them as generalized policy iteration (GPI). GPI is the general idea of two interacting processes revolving around an approximate policy and an approximate value function.
- `bootstrapping: `update estimates on the basis of other estimates. All of these DP methods update estimates of the values of states based on estimates of the values of successor states.


​			
​		
​		











