### [Pooled Variance](https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean)/Scale New Data

- How to calculate pooled variance of two groups given known group variances, means and sample sizes?

  Use the definition of mean,

  <img src="https://ws1.sinaimg.cn/large/006tKfTcgy1fpr5tdh30mj308i03mt8r.jpg" width="200px" />

  and sample variance

  <img src="https://ws4.sinaimg.cn/large/006tKfTcgy1fpr5uz8varj30po04gjru.jpg" width="500px" />

  (the last term in parentheses is the unbiased variance *estimator* often computed by default in statistical software) to **find the sum of squares** of all the data xi. Let's order the indexes i so that i=1,…,n designates elements of the first group and i=n+1,…,n+m designates elements of the second group. Break that sum of squares by group and re-express the two pieces in terms of the variances and means of the subsets of the data:

  <img src="https://ws1.sinaimg.cn/large/006tKfTcgy1fpr5v82vgwj30s6090my5.jpg" width="600px" />

  Algebraically solving this for σ2m+n in terms of the other (known) quantities yields

  <img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fpr5vf53y6j30om03kwev.jpg" width="550px"/>

​	Using the same approach, **μ1:m+n=(nμ1:n+mμ1+n:m+n)/(m+n) **can be expressed in terms of the group means.

- By using above-mentioned formulas, we can use it to scale old data with new data (in TRPO).

  ```python
  n = x.shape[0]
  new_data_var = np.var(x, axis=0)
  new_data_mean = np.mean(x, axis=0)
  new_data_mean_sq = np.square(new_data_mean)
  new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
  self.vars = (((self.m * (self.vars + np.square(self.means))) + (n * (new_data_var + 			new_data_mean_sq))) / (self.m + n) - np.square(new_means))
  self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
  self.means = new_means
  self.m += n
  ```

  ​