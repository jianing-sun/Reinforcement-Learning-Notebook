## Hinge Loss

- In machine learning, the **hinge loss** is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs). For an intended output *t* = ±1and a classifier score y, the hinge loss of the prediction y is defined as:

  <img src="https://ws1.sinaimg.cn/large/006tNc79gy1fppzzop3snj30a601ydft.jpg" width="300px"/>

- Optimization

  The hinge loss is a convex function, so many of the usual convex optimizers used in machine learning can work with it.

  In PPO algorithm, we can just hinge loss as part of our total loss. In that case we are more inclined to smoothed optimization versions, such as quadratically smoothed:

  <img src="https://ws2.sinaimg.cn/large/006tNc79gy1fpq04fsqewj30c802wq2z.jpg"  width="350px" />

  ​