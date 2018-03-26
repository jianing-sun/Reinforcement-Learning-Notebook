## Chapter 9

### On-policy Prediction with Approximation

SUNJIANING   Mar. 4th	HSSL

- The novelty in this chapter is that the approximte value function is represented not as a table but as a parameterized functional form with weight vector $w\in \mathbb{R}^d$
- We will write $\hat v(s,\mathbf{w})\approx v_{\pi}(s)$ for the approximate value of state s given weight vector **w**. For example, $\hat v$ might be a linear function in features of the state, with **w** the vector of feature weights. More generally, $\hat v$ might be the function computed by a multi-layer artificial neural network, with w the vector of connection weights in all the layers.
- When a single state is updated, the change gnerealizes from the state to affect the values of many other states. Such *generalization* makes the learning potentially more powerful but also potentially more difficult to manage and understand.
- Prediction: update to an estimated value function that shift its value at particular states toward a "back-up value", or *updated target* for that state. It is natural to interpret each update as specifying an example of the desired input-output behavior of the value function. **Function approximation methods expect to receive examples of the desired input-ouput behavior of the funcion they are trying to approximate**.

#### 9.1 Value-function Approximation

All of the prediction methods covered in this book have been described as **updates to an estimated value function that shift its value at particular states toward a "backed-up value"**, or `update target`, for that state.

The MC update for value prediction is $S_t \mapsto G_t$, the TD(0) update is $S_t \mapsto R_{t+1}+\gamma \hat v(S_{t+1}, \mathbf{w}_t)$, and the n-step TD update is $S_t \mapsto G_{t: t+n}$. In the DP policy-evaluation update, $s \mapsto \mathbb{E}_\pi \big[R_{t+1}+\gamma \hat v(S_{t+1},\mathbb{w}_t)\big|S_t=s\big]$, an arbitrary state s is updated, whereas in the other cases the state encountered in actual experience, $S_t$, is updated.

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fp1a0alxlaj313c0hwacg.jpg" width="500px"/>

#### 9.2 The Prediction Objective ($\overline {VE}$)

With genuine approximation, an update at one state affects many others, and it is not possible to get the values of all states exactly correct. By assumption we have far more states than weights, so making one state’s estimate more accurate invariably means making others' less accurate.

We are obligated then to say which states we care most about. We mush specify a `state weighting` or distribution $\mu(s)\geq 0, \sum_s\mu(s)=1$, representing how much we care about the error in each state s.

By the error in a state s we mean the square of the difference between the approximate value $\hat v(s, \mathbf{w})$ and the true value $v_\pi(s)$. Weighting this over the state space by $\mu$, we obtain a natural objective function, the `MeanSquared Value Error`, denoted $\overline{VE}$:

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1fp1a758d0ij30jw04cglv.jpg" width="400px"/>

Often $\mu(s)$ is chosen to be the fraction of time spent in s.

Under on-policy training this is called the `on-policy distribution`;

#### 9.3 Stochastic-gradient and semi-gradient Methods

SGD methods are among the most widely used of all function approximation methods and are particularly well suited to online reinforcement learning.

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fp1andta64j312c0cemzr.jpg)

we do not seek or expect to find a value function that has zero error for all states, but only an approximation that balances the errors in different states.

![](https://ws3.sinaimg.cn/large/006tKfTcgy1fp1auw8t0bj31hq0fiwhn.jpg)

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fp1btmnujaj316a0hiad4.jpg)

Bootstrapping methods are not in fact instances of true gradient descent (Barnard, 1993). They take into account the effect of changing the weight vector $\mathbf{w}_t$ on the estimate, but ignore its effect on the target. They include only a part of the gradient and, accordingly, we call them `semi-gradient methods`.

`state aggregation` is the simplest kind of VFA.

- States are partitioned into disjoint subsets (groups)
- One component of **w** is allocated to each group

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1fp1bhkikzwj30su09s0tp.jpg" width="500px" />

#### 9.4 Linear Methods

One of the most important special cases of function approximation is that in which the approximate funcion, $\hat v(\centerdot , \mathbf{w})$, is a linear function of the weight vector. Corresponding to every state s, there is a real-valued vector $\mathbf{x}(s)\doteq (x_1(s), x_2(s),…, x_d(s))^T$, with the same numnber of components as **w**. Linear methods approximate state-value function by the inner product between **w** and $\mathbf{x}(s)$:

$$\hat v(s, \mathbf{w})\doteq \mathbf{w}^T\mathbf{x}(s)\doteq\sum\limits_{i=1}^d w_ix_i(s)$$			
In this case the approximate value function is said to be `linear in the weights`, or `simply linear`.

Thus, in the linear case the general SGD update reduces to a particularly simple form:

![](https://ws3.sinaimg.cn/large/006tKfTcgy1fp1bqx7396j30jc02qaa6.jpg)			
True SGD will converge to a local minimum of the error objective in linear VFA, there is only one minimum: local = global. In the linear case there is only one optimum (or, in degenerate cases, one set of equally good optima), and thus any method that is guaranteed to converge to or near a local optimum is automatically guaranteed to converge to or near the global optimum. For 

The semi-gradient TD(0) algorithm presented in the previous section also converges under linear function approximation, but this does not follow from general results on SGD; a separate theorem is necessary. The weight vector converged to is also not the global optimum, but raather a point near the local optimum. 

The update at each time t is:

$\mathbf{w}_{t+1}\doteq \mathbf{w}_t+\alpha \big(R_{t+1}+\gamma \mathbf{w}_{t}^T\mathbf{x}_{t+1}-\mathbf{w}_t^T\mathbf{x}_t\big)\mathbf{x}_t$			
​	  $=\mathbf{w}_t+\alpha \big(R_{t+1}-\mathbf{x}_t(\mathbf{x}_{t}-\gamma \mathbf{x}_{t+1})^T\mathbf{w}_t\big)$	

where here we have used the notrational shorthand $\mathbf{x}_t=\mathbf{x}(S_t)$. Once the system has reached steady state, for any given $\mathbf{w}_t$, the expected next weight vector can be written:

$\mathbb{E}\big[\mathbf{w}_{t+1}|\mathbf{w}_t\big]=\mathbf{w}_t+\alpha(\mathbf{b}-\mathbf{A}\mathbf{w}_t)$

where,

![](https://ws2.sinaimg.cn/large/006tKfTcgy1fp1cfck3sbj30yi02waac.jpg)			
![](https://ws1.sinaimg.cn/large/006tKfTcgy1fp1cfk5mjoj31gm07qq45.jpg)		
​				
This quantity is called the `TD fixed point`. In fact linear semi-gradient TD(0) converges to this point. 

At the TD fixed point, it has also been proven that the $\overline{VE}$ is within a bounded expansion of the lowest possible error:

$\overline{VE}(\mathbf{w}_{TD})\leq \frac{1}{1-\gamma}\min\limits_{\mathbf{w}}\overline{VE}(\mathbf{w})$		

#### 9.5 Feature Construction for Linear Methods

-  All fast learning is linear learning
  - the original perception
  - the least-mean-square (LMS) algorithm
  - SVMs
- Even in deep learning, learning is linear in the critical last layer

#### 9.5.3 Coarse Coding

Consider a task in which the natural representation of the state set is a **continuous** two-dimensional space. One kind of representation for this case is made up of features corresponding to circles in state space, If the state is inside a circle, then the corresponding feature has the value 1 and is said to be `present`; otherwise the feature is 0 and is said to be `absent`. This kind of 1-0-valued feature is called a `binary feature`. 

Given a state, which binary features are present indicate within which circles the state lies, and thus coarsely code for its location. Representing a state with features that overlap in this way is know as `coarse coding`.

![](https://ws4.sinaimg.cn/large/006tKfTcgy1fp1d5kf9wtj31ge0lqdl6.jpg)

The width of the receptive fields determines breadth of generalization.

#### 9.5.4 Tile Coding





### Conclusion

- Value-function approximation by stochastic gradient descend enables RL to be applied to arbitrarily large state spaces
- Most algorithms just carry over the Targets from the tabular case
- With bootstrapping (TD), we don't get true gradient descent methods
  - this complicates the analysis
  - but the lienar, on-policy case is still guaranteed convergent
  - and learning is till much faster
- For continuous state spaces, coarse/tile coding is a good strategy
- For ambitious AI, artificial neural networks are an interseting strategy


​		
​	































​	