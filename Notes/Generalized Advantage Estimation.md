## [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)

### 1. Introduction

- Credit assignment problem: long time delay between actions and their positive or negative effect on rewards.
- Value functions offer an elegant solution to the credit assignment problem by estimating the goodness of an action before the delayed reward arrives. 

### 2. Preliminaries

- Policy gradient methods maximize the expected total reward by repeatedly estimating the gradient $g:= \nabla_\theta\mathbb{E}[\sum_{t=0}^\infty r_t]$. 

- General form of policy gradient:

  ![](https://ws1.sinaimg.cn/large/006tKfTcgy1fpnlqw1z03j31540kgdjw.jpg)

- The choice $\Psi=A^\pi (s_t,a_t)$ yields almost the lowest possible variance, though in practice, the advantage function is not known and must be estimated. This statement can be intuitively justified by the following interpretation of the policy gradient: a step in the policy gradient direction should increase the probability of better-than-average actions and decrease the probability of worse-than-average actions.

- The advantage functionm ensures whether or not the action is better or worse than the policy's default behavior.

- Bellman residual terms: the Bellman residual is the difference between the two sides of the equation. This is slightly different from the TD error. The TD error is the difference between the two sides of the equation without the expected value, evalutated after just a single transition for (x,u). So the Bellman residual is the expected value of the TD error. The TD error is the Bellman residual plus zero-mean noise.

- Here $\gamma$ used to reduce variance by downweighting rewards corresponding to delayed effects. This parameter is not treated as the discounted factor like in MDPs, but it be treated as a **variance reduction parameter in an undiscounted problem.** 

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fpnm39xh3cj31280aignb.jpg)

- $\gamma-just$ estimator of the advantage function $\hat A_t$: an estimator that does not introduce bias when we use it in place of $A^{\pi,\gamma}$

  ![mage-20180323204446](/var/folders/gn/ryfdjg7537z8w1tkpnm2np5r0000gn/T/abnerworks.Typora/image-201803232044465.png)

### 3. Advantage Function Estimation

- Produce an accurate estimat $\hat A_t$ of the discounted advantage function $A^{\pi, \gamma}(s_t, a_t)$
- Let V be an approximate value function. Define $\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$. If we have the correct value function $V=V^{\pi, \gamma}$, then it is a $\gamma-just$ advantage estimator(unbiased) of $A^{\pi, \gamma}$.
- Taking the sum of k of these $\delta$ terms:

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fpnmdb6l6vj30xu0b8jss.jpg" width="700px"/>

- The generalized advantage estimator GAE($\gamma, \lambda$) is defined as the **exponentially-weighted average **of these k-step estimators:

  <img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fpnmem5m78j30yy0dk0ul.jpg" width="700px" />

- Hence, this GAE has two separate parameters $\gamma$ and $\lambda$, both of which contribute to the bias-variance tradeoff when using an approximate value function. However, they serve different purposes and work best with different ranges of values. γ most importantly determines the scale of the value function V π,γ , which does not depend on λ. Taking γ < 1 introduces bias into the policy gradient estimate, regardless of the value function’s accuracy.

- **Key formula of this paper:**

  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpnmik6jbkj30ze03gmxn.jpg)

### 5. Value Function Estimation

- compute an approximate solution to the trust region problem using the conjugate gradient algorithm. Specifically, solving the quadratic program:

  ![](https://ws4.sinaimg.cn/large/006tKfTcgy1fpnmle3ycwj314i0e6ae7.jpg)

### 6. Algorithm and pseudocode

- Select TRPO as the Policy Optimzation Algorithm, TRPO updates the policy by approximately solving the following constrained optimization problem each iteration:

  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpnmprhc98j30xq0aaq4h.jpg)

- Pseudocode:

  ![mage-20180323210220](/var/folders/gn/ryfdjg7537z8w1tkpnm2np5r0000gn/T/abnerworks.Typora/image-201803232102209.png)

### 7. Conclusion

GAE can be used for a varitey of policy optimization algorithm, in this paper, the authors combine GAE with TRPO (Trust Region Policy Optimization). The generalized advantage estimator has two parameters $\gamma, \lambda$ which adjust the bias-variance tradeoff. The author combine this GAE idea with trust region policy optimization (Eq. 31) and a trust region algorithm that used to optimize a value function (Eq. 30.), both represented by neural networks. 



SUNJIANING  March 23









