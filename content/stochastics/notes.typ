#import "../format.typ": *

#show: notes()[Stochastic Equations for Finance and Biology]

= Background on probability

We have variable $X$ that follows distribution $X tilde P_X (x)$ (pdf). Then,

$
  P_X (x) dif x = PP [X in (x, x + dif x) ] >= 0
$

$
  integral_(-oo)^oo P_X (x) dif x = 1 quad ("cdf")
$

$
  EE[X^m] = integral_(-oo)^oo P_X (x) x^m dif x
$

We have moments:

$
  mu_m (X) = EE[ (X - EE[X])^m ]
$

where $mu_1 (x) = 0$, $mu_2 (x) = VV[x] = EE[(X - EE[x])^2] = EE[X^2] - EE^2 [X]$, $mu_3$ is skewness, $mu_4$ is kurtosis.

*Note*: Variables are capital letters (put lines to  clarify), probabilities is rr face $PP$.

Couple more things:

- Markoff's theorem:
$ PP[X >= a] <= EE[X] / a, quad X >= 0, a > 0 $

- Chebychev inequality:
$ PP[abs(X - EE[X]) >= b^] <= VV[x] / b^2, quad X >= 0, b > 0 $

- Jensen's inequality: Given $phi$ convex (smily),
$
  phi(EE[x]) <= EE[phi(x)]
$

- Cauchy Swartz inequality:
$
  EE[X Y]^2 <= EE[X^2] + EE[Y^2]
$

New part. Given a function

$
  f(x) = cases(lim_(abs(x) -> oo) f(x) = 0, integral_(-oo)^oo f^2 (x) dif x < oo)
$

we can take Fourier transform:

$
  cases(
    cal(F)[f](k) = tilde(f)(k) = integral_(-oo)^oo f(x) e^(-i k x) dif,
    cal(F)^(-1)[tilde(f)(k)] = f(x) = 1/(2pi) integral_(-oo)^oo integral_(-oo)^oo tilde(f)(k) e^(i k x) dif k
  )
$

And what's crucial is the _characteristic function_ (proportional to the inverse Fourier transform):

$
  G_X (k) = integral_(-oo)^oo P_X (x) e^(i k x) dif x
$

Remember normal distribution:

$
  X tilde cal(N)[mu, sigma^2] "iff" P_X (x) = 1/sqrt(2 pi sigma^2) exp(- (x-mu)^2/(2sigma^2))
$

and the standard normal distribution $cal(N)[0, 1]$, where $X = mu + sigma cal(N)[0, 1]$.

Also the log-normal distribution:
$
  eta = cal(N)[m, sigma^2], quad X = e^mu, quad log X tilde cal(N)[mu, sigma^2]
$

Ok, so characteristic function has the property:

$
  G_X (k) = EE[e^(i k x)]
$

and what's nterresting here is that given $Y = alpha X$ then

$
  G_Y (k) = G_X (alpha k)
$

since

$
  G_Y (k) & = EE[e^(i k Y)] \
          & = EE[e^(i alpha k X)] \
          & = G_X (alpha k)
$

There's an interesting relation with the moments and the characteristic function. If we take Taylor series at $0$,

$
  & e^eta = 1 + eta + eta^2/2! + ... \
  => & G_X (k) && = integral_(-oo)^oo P_X (x) sum_(m = 0)^oo (i k)^m / m! x^m dif x \
  & && = sum_(m = 0)^oo (i k)^m / m! integral_(-oo)^oo P_X (x) x^m dif x \
  & && = sum_(m = 0)^oo (i k)^m / m! EE[X^m] \
  & && = sum_(m = 0)^oo k^m / m! [(partial^((m)) G_X (k))/(partial k^m)]_(k = 0) \
$

where we used

$
  (partial^((m)) G_X (k))/(partial k^m) |_k=0 = -i^m EE[x^m]
$

This gives the following result:

$
  G_(cal(N)[mu, sigma^2]) (k) = exp(i mu k - (sigma^2 k^2)/2)
$

== Multi-distributions

Take vector of variables $arrow(X) = (X_1, X_2, X_3, ..., X_n)$. Then we have probability distribution $P_arrow(X)$ in the sense that:

$
  P_arrow(X) (x_1, x_2, ..., x_n) dif x_1 dif x_2 ... dif x_n = PP[X_1 in (x_1, x_1 + dif x_1), ..., X_n in (x_n, x_n + dif x_n)]
$

So now, the characteristic function becomes

$
  G_arrow(X) (arrow(k)) = integral_(-oo)^oo ... integral_(-oo)^oo P_arrow(X) (x) e^(i arrow(k) dot arrow(x)) dif x_1 ... dif x_n = EE[e^(i arrow(k) arrow(X))]
$

which is a complex random variable.

As an aside, two random variables are independent iff

#let indep = $tack.t.double$

$ X_1 indep X_2 <=> P_(X_1, X_2) (x_1, x_2) = P_X_1 (x) P_X_2 (x_2) $

So applying this to the characteristic function:

$
  G_(X_1, X_2) (k_1, k_2) & = EE[e^(i arrow(k) dot arrow(X))] \
  & = integral_(-oo)^oo integral_(-oo)^oo P_(X_1, X_2) (x_1, x_2) e^(i k_1 x_1 + i k_2 x_2) dif x_1 dif x_2 \
  "(Fubini's)" quad & = integral_(-oo)^oo P_(X_1) (x_1) e^(i k_1 x_1) dif x_1
  integral_(-oo)^oo P_(X_2) (x_2) e^(i k_2 x_2) dif x_2 \
  & = G_X_1 (k_1) dot G_X_2 (k_2)
$

The covariance is

#let Cov = "Cov"

$
  Cov[X_1, X_2] = EE[(X_1 - EE[X_1]) dot (X_2 - EE[X_2])]
$

And we have that if $X_1 indep X_2$ then $Cov[X_1, X_2] = 0$, but not the other way around!!

== Probabilities of functions of random variables

Take variable $X tilde P_X (x)$, function $f(x)$ and let $Y = f(X) tilde P_Y (y)$ and not being $P_X (f(x))$ nor $f(P_X (x))$. We want to figure out $P(Y)$, which can be written as

$
  P_Y (y) dif y & = PP[f(X) in (y, y + dif y)] \
                & = integral_(A(y)) P_X (x) dif x \
                & = integral_(-oo)^oo P_X (x) delta[f(x) - y] dif x
$

but it is very hard to find the pdf for a function of a random variable in general. But it's much more tracktable to find the _characteristic_ of a function of a random variable. Namely,

$
  G_Y (k) & = integral_(-oo)^oo P_Y (y) e^(i kappa y) \
  & = integral_(-oo)^oo (integral_(-oo)^oo P_X (x) delta [f(x) - y] dif x) e^(i k y) dif y \
  "(Foubini)" & = integral_(-oo)^oo P_X (x) (integral_(-oo)^oo e^(i k y) P_X (x) delta [f(x) - y] dif y) dif x \
  & = integral_(-oo)^oo P_X (x) e^(i k f(x)) dif x \
  & = EE[e^(i k f(X))]
$

We can get, if we get $Y = f(X_1, X_2) = X_1 + X_2$, then $G_Y (k) = G_(X_1, X_2) (k, k)$, and also assuming $X_1 indep X_2$, then

$
  G_(X_1 + X_2) (k) & = integral_(-oo)^oo integral_(-oo)^oo P_(X_1, X_2) (x_1, x_2) e^(i k(x_1 + x_2)) dif x_1 dif x_2 \
  & = integral_(-oo)^oo P_(X_1) (x_1) e^(i k x_1) dif x_1 dot integral_(-oo)^oo P_(X_2) (x_2) e^(i k x_2) dif x_2 \
  & = G_X_1 (k) dot G_X_2 (k)
$

== Convergence and Law of Large Numbers

There are many ways to define limits in probability. One is

$
  { X_n } ->_(n -> oo) X
$

which is called the _p-limit_:

#let plim = $op("plim", limits: #true)_(n -> oo)$

$
  plim X_n = X <=> forall epsilon > 0, lim_(n -> oo) PP[abs(X_n - X) >= 0] = 0
$

Then we can state the Law of Large Numbers: Given $X$ and $VV[X] = sigma^2 < oo$, then

$
  overline(X)_n = 1/n sum_(j=1)^n X_k quad {X_j} "iid" \
  plim overline(X)_n = EE[X]
$

#let mean = overline

This is because, given an $epsilon > 0$,
$
  lim_(n -> oo) PP[abs(mean(X)_n - EE[X]) >= epsilon > 0] & <= VV[abs(mean(X)_n)]/epsilon^2 \
  & = VV[mean(X)_n]/epsilon^2 \
  & = 1/epsilon^2 VV[1/n sum_(j=1)^n X_j] \
  & = 1/(epsilon^2 n^2) sum_(j=1)^n VV[X_j] \
  & = (n VV[X])/(epsilon^2 n^2) \
  & = (n VV[X])/(epsilon^2 n^2) \
  & = VV[X]/epsilon 1/n \
  & -> 0
$

== Central limit theorem

Given a collection of variables ${X_j}_0^n$ iid, where $EE[X] = 0$ and $VV[X] < oo$, as $N -> oo$,

$
  (sum_(j=1)^n X_j) / sqrt(n) --> cal(N)[0, sigma^2]
$

*Xoaquin's proof*:

Let ${ X_k }$ be $N$ iid samples, such that $EE[X_k] = 0$ and variance $VV[X] < oo$.

= Brownian motion

*(Notes from García Palacios)*

The point is saying that there is a random variable $Delta$ which is the amount of motion in a small time $tau$. Let the pdf of $Delta$ be $phi$. We assume it's symmetric, so $phi(-Delta) = phi(Delta)$ which implies $EE[Delta] = 0$. Then, the number of particles at some time is the number of particles that moved from $Delta$ away to the place:

$
  f(x, t + tau) = integral_(-oo)^oo f(x + Delta, t) phi(Delta) dif Delta
$

where we used $phi(-Delta) = phi(Delta)$.

We Taylor expand $f(x, t + tau)$ to get everything evaluated at $tau$, and eventually we reach

$
  (partial f)/(partial t) = D (partial^2 f)/(partial x^2)
$

where $D = VV[Delta]/(2tau)$.

We can solve this using Fourier:

$
  cal(F) [(partial f)/(partial t)] & = cal(F)[ D (partial^2 f)/(partial x^2)] \
  (partial tilde(f))/(partial tau) & = -D k^2 tilde(f) \
                 => tilde(f)(k, t) & = tilde(f)(k, 0) e^(-D k^2 t) \
$

Assuming initial condition $f(x, 0) = delta(x)$, the solution is $tilde(f)(k, 0) = e^(-i k dot 0) = 1$.

So then, the final distribution is

$
  f(x, t) & = 1/(2pi) integral_(-oo)^oo e^(-k^2 D t) e^(i k x) dif k \
  & = 1/sqrt(4pi D t) exp((-x^2)/(4D t))
  & = cal(N) [0, 2D t]
$

== Stochastic Integral (Itô's Integral)

The idea is that

$
  integral_a^b f(t, W_t) dif W_t = lim_(norm(Pi_n) -> 0^+) sum_(j=1)^n f(t_(j - 1), W_t_(j - 1)) (W_t_j - W_t_(j - 1))
$

The Stochastic integral is not a Riemman sum, and can't be visualized in those ways. It is thought of as the change of the weiner process in a small amount of time, and that's it.

Properties:

- $
    integral_a^b (alpha f + beta g) dif W_t = alpha integral_a^b f dif W_t + beta integral_a^b g dif W_t
  $
- $ EE[integral_a^b f(t, W_t) dif W_t] = 0 $
- $
    EE[(integral_a^b f(t, W_t) dif W_t)^2] = EE[integral_a^b f^2(t, W_t) dif W_t]
  $


#proof[
  $
    EE[integral_a^b f(t, W_t) dif W_t] & = EE[lim_(norm(Pi_n) -> 0^+) sum_(j=1)^n f_(j - 1) (W_t_j - W_t_(j - 1))] \
    & = lim_(norm(Pi_n) -> 0^+) sum_(j=1)^n EE[f_(j - 1) (W_t_j - W_t_(j - 1))] \
    & = lim_(norm(Pi_n) -> 0^+) sum_(j=1)^n EE[f_(j - 1)] cancel(EE[(W_t_j - W_t_(j - 1))]) \
    & = 0 \
  $

  where we used that $W_(j - 1) = W_(j - 1) - W_0$ so it is independent to $W_j - W_(j-1)$.
]

#example[

  Instead of using Itô's integral, we can use

  $
    EE[integral_0^T W_t dif W_t] = EE[lim_(norm(Pi_n) -> 0) sum_(j=1)^n W_j (W_j - W_(j - 1))]
  $

  which is different because we evaluated at a different point in the interval. We are going to see that these integrals have different expectation:

  $
    EE[lim_(norm(Pi_n) -> 0) sum_(j=1)^n W_j (W_j - W_(j - 1))]
    & = lim_(norm(Pi_n) -> 0) sum_(j=1)^n EE[W_j (W_j - W_(j - 1))] \
  $

  and here $W_j$ is not necessarily independent so we can't split it as before (before we have $EE[W_j (W_j - W_(j - 1))] = EE[W_j] EE[W_j - W_(j - 1)]$).

  We can still massage it by completing the square to get something at least:

  $
    EE[W_j (W_j - W_(j - 1))] & = EE[W_j^2 - W_j W_(j - 1)] \
    & = EE[(W_j - W_(j - 1))^2 + W_(j - 1)(W_(j - 1) + W_j)] \
    & = EE[(W_j - W_(j - 1))^2] + EE[W_(j - 1)(W_(j - 1) - W_j)]
  $

  Where now $W_(j-1)$ _is_ independent from $W_(j -1) - W_j$, so we can factor it out and set it to $0$. The integral, then, becomes

  $
    lim_(norm(Pi_n) -> 0) sum_(j=1)^n EE[(W_j - W_(j - 1))^2] = lim_(norm(Pi_n) -> 0) sum_(j=1)^n (t_j - t_(j-1)) = T
  $
]

=== Computing Itô's integral

Now, let's try to compute the integral:

$
  integral_0^T W_t dif W_t
$

The result is going to be a _stochastic process_. It's going to depend on $T$ of course, but it's going to be a random variable in terms of the value of $W_t$ at time $T$.

$
  integral_0^T W_t dif W_t & = lim_(norm(Pi_n) -> 0) sum_(j=1)^n W_(j - 1) (W_j - W_(j - 1)) \
  & = lim_(norm(Pi_n) -> 0) sum_(j=1)^n (W_(j - 1) W_j - W_(j - 1)^2) \
  & = 1/2 lim_(norm(Pi_n) -> 0) sum_(j=1)^n (W_j^2 - W_(j-1)^2) - 1/2 lim_(norm(Pi_n) -> 0) sum_(j=1)^n (W_j - W_(j-1))^2 \
  & = 1/2 (W_T^2 - T) \
$

This shows that the expectation of the stochastic integral is $0$, since $EE[1/2 (W_T^2 - T)] = 1/2 EE[W_T^2] - 1/2 EE[T] = 1/2(T - T) = 0$

#theorem[Itô's Isomestry][
  $
    EE[(integral_a^b f(t, W_t) dif W_t)^2] = EE[integral_a^b f^2 (t, W_t) dif t]
  $
]

#proof[

  $
    EE[(integral_a^b f(t, W_t) dif W_t)^2]
    & = EE[(lim_(norm(Pi_n) -> 0) sum_(j=1)^n f_(j - 1) (W_j - W_(j-1)))^2] \
    & = EE[lim_(norm(Pi_n) -> 0) sum_(j=1)^n f_(j - 1)^2 (W_j - W_(j-1))^2] \ & + 2(lim_(norm(Pi_n) -> 0) sum_(1 <= j < k <= n) f_(j - 1) f_(k-1) (W_j - W_(j-1))(W_k - W_(k - 1)) ] \
    ("because of independence") & = EE[lim_(norm(Pi_n) -> 0) sum_(j=1)^n f_(j - 1)^2 (W_j - W_(j-1))^2] \
    & = lim_(norm(Pi_n) -> 0) sum_(j=1)^n EE[f_(j - 1)^2 (W_j - W_(j-1))^2] \
    & = lim_(norm(Pi_n) -> 0) sum_(j=1)^n EE[f_(j - 1)^2] EE[(W_j - W_(j-1))^2] \
    & = integral_a^b EE[f^2 (t)] dif t \
    & = integral_a^b integral_(-oo)^oo z P_W_t (z) f^2(t) dif z dif t \
    & = EE[integral_a^b f^2(t, W_t) dif t] \
  $
]

If we now assume that Itô's integral follows a normal distribution, we can easily tell that the mean is $0$ and the variance can be determined as:

$
  VV[integral_a^b f(t, W_t) dif W_t] & = EE[integral^2] - EE[integral]^2 \
                                     & = EE[integral_a^b f^2 (t, W_t) dif t] \
                                     & = integral_a^b EE[ f^2 (t, W_t)] dif t \
$

We can now calculate the variance of the following:

$
  integral_0^T W_t dif W_t = 1/2 (W_t^2 - T)
$

which is

$
  VV[1/2 (W_T^2 - T)] = 1/4 VV[W_T^2] = 1/4 (EE[W_T^4] - EE^2[W_T^2]) = 1/2 (3T^2 - T^2) = T^2/2 \
  ("using" VV[W_T] = T "and that" eta ~ cal(N)[0, sigma^2] -> EE[eta^4] = 3 sigma^4)
$

== Stochastic differential equations

#definition[Itô's stochastic differential equation][
  Given initial condition $X_0$, functions $mu : RR^2 -> RR$ and $sigma : RR^2 -> RR$, the _stochastic differential equation_ is defined as:

  $
    dif X_t = mu(t, X_t) dif t + sigma (t, X_t) dif W_t
  $

  The solution is defined as $X_t = f(t, W_t)$.
]

We interpret that equation as

$
  integral_0^T dif X_t = X_T - X_0 = integral_0^T mu(t, X_t) dif t + integral_0^T sigma(t, X_t) dif W_t
$

Another name for Itô's DEs are _Itô's drift-diffusion_ differential equations, which is arguably a better name.

*Sufficient conditions*:
- *Existence*: Growth is at most linear
  $ abs(mu(t, y)) + abs(sigma(t, y)) <= C (1 + abs(y)) $
- *Uniqueness*: Lipschitz continuity
  $
    abs(mu(t, x) - mu(t, y)) + abs(sigma(t, x) - sigma(t, y)) <= B abs(x - y)
  $

The solution of a SDE is a stochastic process. That is, we get a pdf at every moment in time.

#example[
  Let $Y_t = F(t, X_t)$, so $Y_0 = F(0, X_0)$, given initial condition $X_0$. We want to find some drift and some diffusion, which is what Itô's formula gives. Namely,

  $
    dif Y_t = \
              & underbrace(
                  (
                    (partial F)/(partial t) (t, X_t) + mu (t, X_t) (partial F) / (partial y) (t, X_t) + (sigma^2(t, X_t))/2 (partial^2 F)/(partial y^2) (t, X_t)
                  ), "drift"
                ) dif t \
              & + underbrace(
                  (
                    sigma(t, X_t) (partial F)/(partial y) (t, X_t)
                  ), "diffusion"
                )partial W_t
  $

  If we now assume $Y_t = W_t^2$, we get that $F(t, y) = y^2$. The _underlying_ is the Weiner process. And, by definition, the drift is $mu = 0$ and the diffusion is $sigma = 1$. #todo[????]

  Apparently at the end we get

  $
    dif Y_t = dif t + 2 W_t dif W_t, Y_0 = 0
  $

  to solve this we have to make integrals:

  $
    integral_0^T dif Y_t & = integral_0^T dif t + integral_0^T 2 W_t dif W_t \
    Y_T - Y_0 & = T + 2 integral_0^T W_t dif W_t \
    W_T^2 & = T + 2 integral_0^T W_t dif W_t \
    => integral_0^T W_t dif W_t = 1/2 (W_T^2 - T)
  $
]

#proof[of Itô's rule][
  We want to show

  #todo[Something]

  $
    dif F = (partial F)/(partial t) dif t + (partial F)/(partial y) dif y
  $

  #todo[I got tiwed.]
]

#example[

  Ok, leet's try to find some stuff about $W_t^4$. We set $Y_t = W_t^4$ so that $f(t, x) = x^4$ and the underlying is $W_t$. For $X_t = W_t$ we know that

  $
    dif X_t = dif W_t = 0 dot dif t + 1 dot dif W_t
  $

  i.e., that $mu = 0$ and $sigma = 1$.

  If we compute the partial derivatives of $f$,

  $
      (partial f)/(partial t) & = 0 \
      (partial f)/(partial x) & = 4x^3 \
    (partial f)/(partial x^2) & = 12 x^2 \
  $

  So,

  $
    dif Y_t & = (0 + 0 + 1^2/2 12 W_t^2 ) dif t + 1 dot 4 W_t^3 dif W_t \
            & = 6 W_t^2 dif t + 4 W_t^3 dif W_t quad ("this is an SDE")
  $

  Then, we integrate

  $
    integral_0^t dif Y_s & = Y_t - Y_0 = W_t^4 = 6 integral_0^t W_s^2 dif s + 4 integral_0^t W_s^3 dif W_s \
    EE[W_t^4] & = 6 EE[integral_0^t W_s^2 dif s] + 4 EE[integral_0^t W_s^3 dif W_s] \
    & =^"fubini" 6 integral_0^t EE[W_s^2] dif s
    \ &= 6 integral_0^t s dif s = [3s^t]_0^t = 3t^2
  $

  You can also calculate $EE[W_t^4]$ using the fact that $W_t ~ cal(N)[0, t]$ to get the same result.
]

= Modeling

We can take a simplified general stochastic integral where $f$ is deterministic:

$
  integral_0^T f(t) dif W_t & ~ cal(N)[0, EE[(integral_0^T f(t) dif W_t)^2]] \
                            & ~ cal(N)[0, EE[integral_0^T f^2(t) dif t]] \
                            & ~ cal(N)[0, integral_0^T f^2(t) dif t] \
$

(so we need the $f$ to be in $L^2_((0, T))$).

== Arithmetic Brownian Motion

Also known as _Brownian Motion with drift_. It is defined and solved by

$
  dif X_t & = a dif t + b dif W_t, X_0 \
      X_t & = X_0 + a integral_0^t dif s + b integral_0^t dif W_s \
      X_t & = X_0 + a t + b W_t \
$

This solution is unique, since for uniqueness we need that the drift and diffusion are Lipschitz continuous, and the constants $a$ and $b$ clearly are.

We can also take a look at expectation:

$
  EE[X_t] & = EE[X_0 + a t + b W_t] \
          & = EE[X_0] + a t + b cancel(EE[W_t]) \
          & = EE[X_0] + a t \
$

and the variance:

$
  VV[X_t] & = VV[X_0 + a t + b W_t] \
          & = VV[X_0] + cancel(VV[a t]) + b^2 VV[W_t] \
          & = VV[X_0] + b^2 t \
$

In fact, assuming the initial condition $X_0 = x_0$ is deterministically just a number,
$
  X_t ~ cal(N)[x_0 + a t, b^2 t]
$


== Geometric Brownian Motion

#example[

  Let's randomly calculate $EE[e^(alpha W_t)]$

  $
    EE[e^(alpha W_t)] \
    Y_t = e^(alpha W_t) \
    dif W_t = 0 dif t + 1 dif W_t \
    f(t, xi) = e^(alpha xi) \
    (partial f)/(partial t) = 0 \
    (partial f)/(partial xi) = alpha e^(alpha xi) \
    (partial f)/(partial xi^2) = alpha^2 e^(alpha xi) \
    Y_0 = e^(alpha W_0) = 1
  $

  Using the formula,

  $
    dif Y_t & = (0 + 0 + 1^2/2 alpha^2 e^(alpha W_t)) dif t + 1 alpha e^(alpha W_t) dif W_t \
    & = alpha^2/2 Y_t dif t + alpha Y_t dif W_t
  $

  And solving this,

  $
    Y_t & = 1 + alpha^2/2 integral_0^t Y_s dif s + alpha integral_0^t Y_s dif W_s \
    EE[Y_t] & = 1 + alpha^2/2 EE[integral_0^t Y_s dif s] + 0 \
    & =^"fubini" 1 + alpha^2/2integral_0^t EE[Y_s] dif s \
  $

  This is now an ODE! $EE[Y_t]$ is just a function of $t$, say $E(t)$, so

  $
    E(t) = 1 + alpha^2/2 integral_0^t E(s) dif s
  $

  Even though this one is simple, in general to solve these type of problems you need Leibniz's formula:

  $
    I(x) & = integral_g(x)^f(x) F(x, y) dif y \ & = F(x, f(x)) f'(x) - F(x, g(x)) g'(x) + integral_g(x)^f(x) (dif F)/(dif x) (x, y) dif y
  $

  In our problem, we only have a function in the upper limit,  where $f'(t) = (t)' = 1$, and so

  $&
  E'(t) & = alpha^2/2 E(t) \
  E(t) = E(0) e^(alpha^2/2 t)$

  where $E(0) = 1$, and thus

  $
    EE[e^(alpha W_t)] = e^((alpha^2 t)/2)
  $

  However, this is not the typical way to get geometric brownian motion.
]

#definition[Geometric Brownian Motion][
  $
    dif X_t = mu X_t dif t + sigma X_t dif W_t, quad X_0
  $

  In words, GBM is the process where both drift and diffusion are linear.
]

Checking for uniqueness, again we see it is unique since linear functions are clearly Lipschitz continuous, but let's be thorough:

$
  abs(mu (x - y)) + abs(sigma x - sigma y) = (abs(mu) + abs(sigma)) abs(x - y) <= alpha abs(x - y)
$

holds since $abs(mu) + abs(sigma)$ is a nonnegative constant.

#example[
  Let's calculate $dif Y_t$ for $Y_t = log (X_t)$. We have

  $
    Y_0 = log(x_0) \
    f(t, x) = log x \
    (partial f)/(partial t) = 0, (partial f)/(partial x) = 1/2, (partial^2 f)/(partial x^2) = -1/x^2
  $

  Plugging it into the formula,

  $
    dif Y_t & = (0 + 1/X_t mu X_t + 1/2 sigma^2 X_t^2 (-1/X_t^2)) dif t + 1/X_t sigma X_t dif W_t \
    & = (mu - sigma^2/2) dif t + sigma dif W_t
  $
]

The point is that $Y_t = log(X_t)$ is ABM! Since ABM is a normal, GBM is a log-normal. And with this we can solve for GBM,

$
  Y_t & = Y_0 + (mu - sigma^2/2) t + sigma W_t \
  X_t & = X_0 exp((mu - sigma^2/2) t + sigma W_t)
$


#example[
  An argument. A heuristic one.

  Take GBM:

  $
    dif X_t = mu X_t dif t + sigma X_t dif W_t, X_0
  $

  we want to show that if $X_0 > 0$ then $X_t > 0$ for all $t$. That is, that GBM preserves positivity. The idea is that $X_t$ inherits continuity from the Weiner process, so if at some point $X_r > 0$ and at some other point $X_s < 0$ then there is some moment where $X_tau = 0$. Then, $dif X_t = 0 dif t + 0 dif W_t$ at the time $tau$, so $X_t$ is constant for $t >= tau$ and specifically $0$.
]

GBM is used for many models but one of them is for stock prices.

#example[invest in stocks or invest in bank?][
  Assuming stocks follow GBM, we want to determine $EE[X_t]$, which is

  $
    EE[X_t] & = x_0 e^((mu - sigma^2/2)t) underbrace([e^(sigma W_t)], e^(sigma^2 t)/2) \
    & = x_0 e^(mu t)
  $

  the bank gives you $y(t) = x_0 e^(r t)$. So firstly we need $mu > r$. The thing is checking the variances. $VV[y] = 0$ since it's deterministic, but

  $
    VV[X_t] & = x_0^2 e^((2mu - sigma^2)t) VV[e^(sigma W_t)] \
    & = x_0^2 e^((2mu - sigma^2)t) (EE[e^(2 sigma W_t)] - EE[e^(sigma W_t)]^2) \
    & = x_0^2 e^((2mu - sigma^2)t) (e^(2sigma^2 t) - e^(sigma^2 t)) \
    & = x_0^2 e^((2mu - sigma^2)t) (e^(sigma^2 t) (e^(sigma^2 t) - 1)) \
    & = x_0^2 e^(2mu t) (e^(sigma^2 t) - 1) \
  $
]

== Orstein-Uhlenbeck process

$
  dif X_t = ((alpha - X_t)/tau) dif t + sigma dif W_t, X_0
$

This is similar to GBM, but where the noise is independent of the value. That is, in GBM if the stock is higher the noise is higher, here it is constant.

But more importantly, the first term models regression to the mean. As the process advances in time, it goes to the mean, which is going to be $alpha$, since if $X_t > alpha$ then the drift term pulls $X_t$ down and if $X_t < alpha$ it pulls it up, pulling it to the mean, $alpha$. We also expect that the variance will go down in time, as it goes to the mean.

First, let's solve it:

We take $Y_t = e^(t/tau) X_t$, where $f(t, x) = x e^(t/tau)$. Then,

$
  dif Y_t & = ((1/tau X_t e^(t/tau) + (alpha - X_t)/tau) e^(t/tau) + 0) dif t + sigma e^(t/tau) dif W_t \
  & = alpha/tau e^(t/tau) dif t + sigma e^(t/tau) dif W_t \
  Y_t & = Y_0 + alpha/tau integral_0^t e^(s/tau) dif s + sigma integral_0^t e^(s/tau) dif W_s
$

Therefore, the analytical solution for OU is:

$
  X_t = X_0 e^(-t/tau) + alpha (1 - e^(-t/tau)) + sigma e^(-t/tau) integral_0^1 e^(s/tau) dif W_s
$

#faint[(the integral itself doesn't have a primitive)]

This is a normal since it's deterministic + deterministic + stochastic integral with deterministic integrand (which is a normal) so the result is normal.

*Properties:*

Let's see that the expectation is indeed $alpha$ as $t -> oo$.

$
  EE[X_t] & = EE[X_0 e^(-t/tau) + alpha (1 - e^(-t/tau)) + sigma e^(-t/tau) integral_0^1 e^(s/tau) dif W_s] \
  & = EE[X_0 e^(-t/tau)] + EE[alpha (1 - e^(-t/tau))] + EE[sigma e^(-t/tau) integral_0^1 e^(s/tau) dif W_s] \
  ("as" t -> oo) quad & = alpha \
$

And let's also see that the variance goes to $0$ as $t -> oo$:

$
  VV[X_t] & = VV[X_0 e^(-t/tau) + alpha (1 - e^(-t/tau)) + sigma e^(-t/tau) integral_0^1 e^(s/tau) dif W_s] \
  & = sigma^2 e^(-(2t)/tau) VV[integral_0^1 e^(s/tau) dif W_s] \
  & = sigma^2 e^(-(2t)/tau) (EE[(integral_0^1 e^(s/tau) dif W_s)^2] - EE[integral_0^1 e^(s/tau) dif W_s]^2) \
$

This model is nice for spikes.

This model doesn't preserve positivity, so people thought it couldn't be used for stock prices, but actually the houses went negative at some point so nobody here knows what they're doing I guess.

== Malthusian SDE

We want to model the number of individuals in a population $N(t)$, where

$
  dif N = (b - d) N dif t
$

where $b$ is the probability of birth and $d$ is the probability of death. The resulting equation is the classic exponential:

$
  N(t) = N_0 e^((b-d)t)
$

However, this explodes as time goes on, and doesn't consider extinction _probability_. Let's stochastify the model, then:

$
  dif N_t = r N_t dif t + sigma N_t dif W_t
$

where here we essentially add some noise to the growth rate $r$. This is GBM which we know how to describe:

$
  N_t = N_0 exp((r - sigma^2/2)t + sigma W_t)
$

This case is an example where the initial condition $N_0$ makes sense to be a distribution.

Now, let's examine extinction.

=== Extinction

We want to calculate the probability that the species goes extinct at some $T$ or before, which is

$
  PP[min_(0<t<T) N_t < e]
$

where $e$ is the extinction threshold. Numerically, if we can launch trajectories of the SDE (for instance, using Euler-Maruyama (@sec-euler-maruyama)) we can just count the number of trajectories that go below the extinction threshold over the total number of trajectories.

Notice that we can't look at only the endpoint, we have to look if *at any point* the population crosses the threshold. Numerically, this is actually very hard to do this because between any two discrete samples there could be stochastic variation that crosses the threshold, but we wont bother with that now.

On the other hand, we can solve this analytically, since we just have log-normals, using the reflection principle.

#theorem[Reflection principle][
  $
    PP[min_(0<t<T) W_t < b] = 2 PP[W_T >= b]
  $

  That is, the probability of a Weiner process crossing a barrier at a value $b$ is twice the probability of the process _ending_ above $b$.
]

#proof[
  If a Weiner process reaches $b$, by symmetry it will end up the same above and below $b$. Therefore, $1/2 PP[W_t "crossing" b] = PP[W_t >= b]$, which implies $PP[W_t "crossing" b] = 2 PP[W_t >= b]$.
]

This actually also helps numerically, but in our case we just have to compute $2 PP[N_T < e]$. Given that $N_t$ is GBM, then $X_t = log(N_t)$ is ABM, with $dif X_t = (r - sigma^2/2) dif t + sigma dif W_t$.

Then, we can compute

$
  PP[N_t "crossing" e] = PP[X_t "crossing" log e]
$

== Logistic SDE

A more accurate population model, that takes into account carrying capacity of environment. Namely,
$
  dif N_t = r N_t (1 - N_t/K) dif t + sigma N_t dif W_t.
$

the two absorbing states are $K$ and $0$, so either the poppulation gets extinct, or when it saturates the environment.

How to calculate extinction probability here? Well, the SDE doesn't have closed-form solution


== Random problems

#example[
  Let's solve

  $
    dif X_t = dif t + 2 sqrt(X_t) dif W_t, quad X_0 = x_0 > 0
  $

  we do a change of variable $Y_t = sqrt(X_t)$, so $f(x) = sqrt(x)$ and $Y_0 = sqrt(X_0)$. Then, Itô's lemma gives:

  $
    dif Y_t & = (0 + frac(1, 2 sqrt(X_t)) - 1/2 1/4 X_t^(-3/2) dot 2 sqrt(X_t) ) dif t + 2 sqrt(X_t) 1/(2sqrt(X_t)) dif W_t \
    & = dif W_t
  $

  So $Y_t = W_t + sqrt(x_0)$, and finally

  $
    X_t = (sqrt(x_0) + W_t)^2
  $

  This is another example of a process that preserves positivity.
]


= Numerical methods

Maybe this didn't need a section because we only have one method? And it's not even that sophisticated.

== Euler-Maruyama method <sec-euler-maruyama>

This is like the Euler method, but for stochastics, with a source of randomness.
Namely, we discretize as

$
  && dif X_t & = & X_t r dif t & + & X_t sigma dif W_t \
  ~~> && hat(X)_(k+1) - hat(X)_k & = & hat(X_k) r h & + & hat(X)_k sigma cal(N) [0, h] \
  ==> && hat(X)_(k+1) & = hat(X)_k & + hat(X_k) r h & + & hat(X)_k sigma cal(N) [0, h] \
$

In general, if we have

$
  dif X_t = f(t, X_t) dif t + g(t, X_t) dif W_t
$

we discretize as

$
      hat(X)_0 & = X_0 \
  hat(X)_(k+1) & = hat(X)_k + f(k h, hat(X)_k) h + g(k h, hat(X)_k) sqrt(h) eta_k \
         eta_k & tilde cal(N)[0, 1]
$

=== Errors

There are two main errors we can think of:

#definition[Strong error][
  Difference between trajectories #todo[?]
]

#definition[Weak error][
  When trying to compute $ EE[phi(X_t)] $

  the weak error is $ e_w = EE[phi(X_T)] - EE[phi(hat(X)_T)] $
]

The weak error is more often used, especially in things like finance.

We know that for Euler-Maruyama has linear weak error.

= Feynman-Kac formula

Given, $phi, c : RR^2 -> RR$ and
$
  I_t = phi(t, X_t) e^(integral_0^t c(s, X_s) dif s)
$

and SDE

$
  dif X_t = b(t, X_t) dif t + a(t, X_t) dif W_t
$

we want to find a function such that

$
  I_t = f(t, X_t)
$

We could think that we can use

$
  f(t, y) = phi(t, y) e^(integral_0^t c(s, y) dif s) quad "(WRONG)"
$

but this is incorrect, since for $y = X_t$, the integrand of the exponential has a $X_t$ instead of a $X_s$!

Ok, so let's do it properly. We start with

$
  cal(Z)_t & = integral_0^t c(s, X_s) dif s \
  =>^"Leibniz's" quad dif cal(Z)_t & = c(t, X_t) dif t \
  =>^"Ito" quad Y_t & = e^(integral_0^t c(s, X_s) dif s) \
  & = e^(cal(Z)_t) quad "for" quad f(t, z) = e^z \
  dif Y_t & = (0 + e^(cal(Z)_t) c(t, X_t) + 0) dif t + 0 dif W_t \
  & = c(t, X_t) e^(integral_0^t c(s, X_s) dif s) dif t
$

Now let's try to get $dif I_t$:

$
          I_t & = phi(t, X_t) e^(cal(Z)_t) Y_t \
      dif I_t & = Y_t dif phi + phi dif Y_z \
  phi dif Y_z & = e^(integral_0^t c(s, X_s) dif s) phi(t, X_t) c(t, X_t) \
  Y_t dif phi & = e^(integral_0^t c(s, X_s) dif s) (#todo[a shit ton of terms]) \
      dif I_t & = e^integral (
                  ((partial phi)/(partial t) + b (partial phi)/(partial x) + a^2/2 (partial phi)/(partial x^2) + phi c) dif t + ( ... ) dif W_t
                )
$

The idea to Feynman-Kac is that you have a parabolic equation $alpha u = (partial u)/(partial t)$ and you can find the solution at certain points by simulating random stochastic paths. Also, if you solve the equation you can get some nice expectations needed in stochastics.

Professor says this is the best idea in math of the past 50 years, so it seems to be a big deal.
