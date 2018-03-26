## Chapter 6

### Temporal-Difference Learning

SUNJIANING Mar. 3rd	  HSSL 

- TD learning is a combination of MC ideas and DP ideas. Like MC methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for final outcome.

*question: why TD and DP bootstrap (update invovles an estimate of the value function) whereas MC do not bootstrap? TD and MC do use experience to solve the prediction problem, like given some experience following a policy $\pi$, both methods update their estimate V of $v_\pi$ for the nonterminal states S_t occurring in that experience.*



#### 6.1 TD Prediction

Whereas Monte Carlo methods mush wait until the end of the episode to determine the increment to $V(S_t)$ (only then is $G_t$ known), TD methods need to wait **only until the next time step**. At time t+1 they immediately form a target and make a useful update using the observed reward $R_{t+1}$ and the estimate $V(S_{t+1})$.

MC: $$V(S_t)\leftarrow V(S_t)+\alpha \big[G_t-V(S_t)\big]$$

MC: $$V(S_t)\leftarrow V(S_t)+\alpha \big[G_t-V(S_t)\big]$$

TD: $$V(S_t)\leftarrow V(S_t) + \alpha\big[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\big]$$

`TD(0)/one-step TD: ` the simplest TD method. It makes the update immediately on trainsition to $S_{t+1}$ and receiving $R_{t+1}$.

**In effect, the target for MC update is $G_t$, whereas the target for TD update is $R_{t+1}+\gamma V(S_{t+1})$**.

![](https://ws1.sinaimg.cn/large/006tNc79gy1fp0bsey1jlj31jy0hywhx.jpg)

*problem: even if TD(0) only wait one step forward, in that step how could we compute $V(S_{t+1})$? Does that mean we need to do one step forward again??*

*my answer: at first we initialize our value matrix, then for every episode we will update our V(s) based on V(S')*

Becase TD(0) bases its update in part on an existing estimate, we say that it is a **bootstrapping** method, like DP.

The target is an estimate for both reasonsL it samples the expected valuee and it uses the current estimate V instead of the true $v_\pi$. Thus, TD methods combine the sampling of MC with the bootstrapping of DP.

**backup diagram** for tabular TD(0):

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0cl0kq5lj30500a0dfy.jpg" width="100px"/>

We refer to TD and Monte Carlo updates as sample updates because they involve looking ahead to a sample successor state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then updating the value of the original state (or state–
action pair) accordingly. Sample updates differ from the expected updates of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

`TD error: ` $$\delta_t \doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)$$			
<img src="https://ws1.sinaimg.cn/large/006tNc79gy1fp0h2lete4j31020q4aek.jpg" width="600px"/>

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fp0h2xbxhbj30uw0p2dju.jpg" width="600px"/>

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0h38pnsaj30vm0ouq73.jpg" width="600px"/>

#### 6.2 Advantages of TD Prediction Methods

- TD methods do not require a model of the environment, only experience
- TD methods can by fully incremental
  - Make updates **before** knowing the final outcome
  - Requires less memory
  - Requires less peak computation
- You can learn **without** the final outcome, from incomplete sequences
- Both MC and TD converge

#### 6.3 Optimality of TD(0)

`batch updating: ` updates are made only after processing each complete batch of trianing data

- difference between the estimates found by batch TD(0) and btch MC methods:
  - Batch Monte Carlo methods always find the estimates that minimize mean-squared error on the training set
  - batch TD(0) always finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process.

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0jhwv1a2j30ni07aweq.jpg" width="500px" />

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0ji2vm3sj30gw0cy75b.jpg" width="500px"/>



- The prediction that best matches the training data is V(A)=0:
  - This minimizes the mean-square-error between V(s) and the sample returns in the training set.
  - Under batch training, this is what constance-$\alpha$ MC gets
- TD(0) achieves a dfferent type of optimality, where V(A)=0.75:
  - This is correct for the maximum likelihood estimate of the Markov model generating the data
  - i.e., if we do a best fit Markov model, and assume it is exactly correct, and then compute the predictions
  - This is called the **certainty-equivalence estimate** because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated
  - This is what TD gets

#### 6.4 Sarsa: On-policy TD Control

The first step is to learn an action-value function rather than a state-value function. In particular, for an on-policy method we must estimate $q_\pi(s,a)$ for the current behavior policy $\pi$ and for all states s and actions a.

In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state–action pair to state–action pair, and learn the values of state–action pairs.	

Formally these cases are identical: they are both Markov chains with a reward process. The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm for action values:

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp0juwtoudj30xy02wq35.jpg)

This rule uses every element of the quintuple of events, **(St, At, Rt+1, St+1, At+1)**, that make up a transition from one state–action pair to the next.

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0jvrukqij304808o0ss.jpg" width="100px"/>

It is straightforward to design an on-policy control algorithm based on the Sarsa prediction method. As in all on-policy methods, we continually estimate qπ for the behavior policy π, and at the same time change π toward greediness with respect to qπ. 

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp0jxccau1j31gw0h442e.jpg)

#### 6.5 Q-learning: Off-policy TD Control

Defined by:

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp0k0rege9j30ys02ujro.jpg)

![](https://ws4.sinaimg.cn/large/006tNc79gy1fp0k2c31xkj31h20fi77p.jpg)

![](https://ws4.sinaimg.cn/large/006tNc79gy1fp0k4qmnv0j30xo0c63zw.jpg)

*question: why Q-learning is off-policy TD control?*

*my answer: In Sarsa, both current action A and next action A' are using policy derived from Q, whereas in Q-learning, it takes next action based on the max Q value, not on the same policy.*

#### 6.6 Expected Sarsa

Consider the learning algorithm that is just like Q-learning except that instead of the maximum over next state–action pairs it uses the expected value, taking into account how likely each action is under the current policy. That is, consider the algorithm with the update rule:

![](https://ws2.sinaimg.cn/large/006tNc79gy1fp0k6c0kobj313m05s0tj.jpg)

Expected Sarsa is more complex computationally than Sarsa but, in return , it eliminates the variance due to the random selection of $A_{t+1}$. 

- Off-policy Expected Sarsa

Expected Sarsa generalizes to arbitrary behavior policies $\mu$. It includes Q-learning as the special case in which $\pi$ is the greedy policy.

#### 6.7 Maximization Bias and Double Learning

Double Q-learning: divides the time steps in two, perhaps by flipping a coin on each step. If the coin comesup heads, the update is:

![](https://ws2.sinaimg.cn/large/006tNc79gy1fp0l2kjsibj314w02yq3b.jpg)

If the coin comes up tails, then the same update is done with Q1 and Q2 switched, so that Q2 is updated. The two approximate value functions are treated completely symmetrically. The behavior policy can use both action-value estimates. For example, an ε-greedy policy for Double Q-learning could be based on the average (or sum) of the two action-value estimates.

![](https://ws3.sinaimg.cn/large/006tNc79gy1fp0l0vu09nj31d60k60ww.jpg)

#### 6.8 Games, Afterstates, and Other Special Cases

A conventional state-value function evaluates states in which the agent has the option of selecting an action, but the state-value function used in tic-tac-toe evaluates **board positions** after the agent has made its move.

These are called `afterstates`.			
<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp0l8k9a04j30pa0e6my0.jpg" width="500px" />		

In such cases the position–move pairs are different but produce the same “afterposition,” and thus musthave the same value. A conventional action-value function would have to separately assess both pairs,whereas an afterstate value function would immediately assess both equally. Any learning about theposition–move pair on the left would immediately transfer to the pair on the right.

#### 6.9 Summary

As usual, we divided the overall problem into a prediction problem and a control problem. TD methods are alternatives to MC methods for solving the prediction problem.

Common points: They will be able to process experience on-line, with relatively little computation, and they will be driven by TD errors.


​			
​		
​			

​		
​				
​		
​				
​		

​	
​		
​			
​		
​	















​	







