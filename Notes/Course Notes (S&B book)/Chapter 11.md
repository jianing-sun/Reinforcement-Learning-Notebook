## Chapter 11

### Off-policy Methods with Approximation

SUNJIANING Mar. 5th  HSSL

- In off-policy learning we seek to learn a value function for a target policy $\pi$, given data due to a different behaviour policy b. In the prediction case, both policies are static and given, and we seek to learn either state values $\hat v\approx v_\pi$ or action values $\hat q \approx q_\pi$. In the control case, action values are learned, and both policies typically change during learning — $\pi$ being the greedy policy with respect to $\hat q$, and b being something more exploratory such as the $\varepsilon$-greedy policy with respect to $\hat q$.
- *Importance sampling* can address the challenge with the target of the update in tabular case. 

#### 11.1 Semi-gradient Methods

