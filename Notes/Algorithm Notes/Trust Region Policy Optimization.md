## [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

#### 1. Preliminaries

- Let η(π) denote the expected return of π,

  <img src="https://ws2.sinaimg.cn/large/006tNc79gy1fppx9dkyrsj30lk05s0t5.jpg" width="500px"/>

  Let ρπ be the (unnormalized) discounted visitation frequencies,

  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fppxf23zf1j30rk02wt8u.jpg)			

- We collect data with πold. Want to optimize some objective to get a new policy π. A useful identity from Kakade & Langford(2002) expresses the expected return of another policy π ̃ in terms of the advantage over π, accumulaterd over timesteps:

  <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fppxdwmvdtj30qc04qaag.jpg" width="600px"/>

  Rewrite Equation (1) with a sum over states instead of timesteps:

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppxk04yy4j30hm02kaa8.jpg" width="500px"/>

  This equation implies that any policy update π → π ̃ that has a nonnegative expected advantage at every state s, i.e.,  a π ̃(a|s)Aπ(s,a) ≥ 0, is guaranteed to increase the policy performance η, or leave it constant in the case that the expected advantage is zero everywhere.		

  Let η(π) denote the expected return of π,	

- Surrogate Loss Function: define Lπold (π) to be the "surrogate objective" that ignores change in state distribution:

  <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fppxmw09hoj30qo034aac.jpg" width="550px" />

- Matches to first order for parameterized policy:

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppxnojobsj30ni04wweu.jpg" width="550px"/>

  Equation (4) implies that a sufficiently small step πθ0 → π ̃ that improves Lπθold will also improve η, but does not give us any guidance on how big of a step to take.

#### 2. Monotonic Improvement Guarantee for General Stochastic Policies

- Improvement Theory: bound the difference between Lπold (π) and η(π)

  <img src="https://ws2.sinaimg.cn/large/006tNc79gy1fppxt6jwcyj30mi05agm4.jpg" width="500px" />

- Monotonic imporvement guarantedd (MM algorithm)

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppxu6nn20j30kw0esjtg.jpg" width="400px"/>

#### 3. Optimization of Parameterized Policies

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fppxxlg4g6j30s2084abw.jpg" width="500px"/>

- In practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small. One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint:

  <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fppxyc5nanj30m804sdgc.jpg" width="500px" />

- This problem imposes a constraint that the KL divergence is bounded at every point in the state space. While it is motivated by the theory, this problem is impractical to solve due to the large number of constraints. Instead, we can use a heuristic approximation which considers the average KL divergence:

  <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fppy0cun5vj30pc02iglt.jpg" width="500px" />

  <img src="https://ws4.sinaimg.cn/large/006tNc79gy1fppy1x8uvoj30q405a3z9.jpg" width="550px" />

- Match the objective and constraint functions in Monte Carlo simulation:

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppy2xlaztj30r405wab1.jpg" width="550px"/>

#### 4. Practical Algorithms

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fppy4tip32j30rq0gutcl.jpg" width="500px" />

#### 5. Approximating Factored Policies with Neural Networks (Appendix D)

- The policy, which is a conditional probability distribution πθ(a|s), can be parameterized with a neural network. This neural network maps (deterministically) from the state vector s to a vector μ, which specifies a distribution over action space. Then we can compute the likelihood p(a|μ) and sample a ∼ p(a|μ).

- For our experiments with continuous state and action spaces, we used a Gaussian distribution, where the covariance matrixwas diagonal and independent of the state. A neural network with several fully-connected (dense) layers maps from the input features to the mean of a Gaussian distribution. A separate set of parameters specifies the log standard deviation ofeach element. More concretely, the parameters include a set of weights and biases for the neural network computing themean, {Wi, bi }Li=1 , and a vector r (log standard deviation) with the same dimension as a. Then, the policy is defined by the normal distribution:

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppy85ghzpj30ok020t8z.jpg" width="500px" />

  Here, μ = [mean, stdev].



