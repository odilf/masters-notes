= Problem

*Is $I_t = integral_0^t W_s dif s$ normally distributed?*

= Solution 1

By writing the definition of the Rieman integral:

$
  I_t & = lim_(norm(Pi_n) -> 0) sum_(j = 1)^n sum_(j=1)^n W_(j-1) (t_j - t_(j-1)) \
  & = W_0 (t_1 - T_0) + W_1 (t_2 - t_1) + W_2 (t_3 - t_2) + ... + W_(n-1) (t_n - t_(n+1)) \
  & = -t_0 W_0 - t_1 (W_1 - W_0) - t_2 (W_2 - W_1) - ... - t_n W_(n-1)
$

All the segments $W_0$, $W_1 - W_0$, $W_2 - W_1$ are non-overlapping and therefore independent. However, the last one is problematic, since it's not independent. However, we can write $W_(n-1)$ as a telescopic sum, namely:

$
  W_(n - 1) = sum_(k=1)^(n-1) (W_k - W_(k-1))
$

where the only Wiener process property we use is that $W_0 = 0$. Then, continuing,
$
  I_t & = -t_0 W_0 - t_1 (W_1 - W_0) - t_2 (W_2 - W_1) - ... - t_n W_(n-1) \
  & = -t_0 W_0 - sum_(k=1)^(n-1) t_k (W_k - W_(k-1)) - t_n W_(n-1) \
  & = cancel(-t_0 W_0) - sum_(k=1)^(n-1) t_k (W_k - W_(k-1)) - t_n sum_(k=1)^(n-1) (W_k - W_(k-1)) \
  & = - sum_(k=1)^(n-1) (t_k + t_n) (W_k - W_(k-1)) \
$

And here all segments are non-overlapping and thus independent. Since a linear combination of independent normals is normal, $I_t$ is normal.

= Solution 2

We can do change of variable $X_t = t W_t$. Then, using Itô's lemma we get:

$
  dif (t W_t) = t dif W_t + W_t dif t \
  [s W_s]0^t = integral_0^t s dif W_s + integral_0^t W_s dif s \
  integral_0^t W_s dif s = t W_t - integral_0^t s dif W_s
$

Now, $t W_t$ is normal and $integral_0^t s dif W_s$ is normal too (since it's a stochastic integral with deterministic integrand), and their subtraction is gaussian *if the gaussians are independent*. And we can't even show that they are independent using covariance because even if covariance is $0$ variables can be dependent.
