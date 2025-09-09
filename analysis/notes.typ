#import "../format.typ": *

#show: notes(
  subtitle: [Approximating functions by "simpler" other functions of some specified type.]
)[Advanced Methods in Applied Analysis]

= Motivating examples

== Polynomials and Taylor

*Objective*: Determine a polynomial $p$ such that $|e^x - p(x)| <= 10^(-6)$.

Idea: use the n#super[th] order Taylor approximation of $f$ around $x = 0$.

$ f(x) = e^x $
$ P_n (x) = sum_(k=0)^n (f^((k)) (x)) / (k!) x^k $

#theorem("Taylor's")[
  Given an interval $I subset RR$ around $a$ and $f: I -> RR$ being $C^infinity$.
  If a constant $C_k > 0$ exists such that $|f^((k))(x)| <= C_k$ (i.e., any point of any $k^"th"$ derivative of $f$ is bounded by $C_k$). 

  Then, the residual of the polynomial $P_n (x)$ around $x = a$ is bounded by
  $ |f(x) - P_n (x)| <= C_k (x - a)^(k + 1) / (k + 1)! $
]

Applying the above to the previous, in the interval $x in [-1, 1]$ the upper bound is $C_k = e^1 = e$ (since all derivatives are also $e^x$). Therefore:

$ |e^x - P^n (x)| <= e (|x|^(n+1)) / (n+1)! <= e / (n + 1)! $

In the last step we use the fact that $|x|^(n+1) <= 1$ because $x in [-1, 1]$.

To solve it we can test values of $n$, and we arrive that for $n = 8$, $e / (n+1)! < 10^(-6)$. So we need 8 terms.

#theorem("Taylor series")[
  $I subset RR$ interval, $f: I->R$ is $C^inf$, $x_0 in I$ and assume
]

#corollary[
  Assume same conditions as before.

  Given any $epsilon > 0$
]

== Periodic functions and Fourier

Given a $2 pi$-periodic function $f$, give an approximation in the form of a "trigonomic polynomial", i.e.: $ sum_(k=0)^n (a_k cos(k x) + b_k sin(k x)) $

If $f$ is "nice" then we can use the theory of Fourier series.

#theorem("Fourier series")[
  Suppose $f : R -> R$ is $2 pi$ periodic and that $f$ is differentiable and continuous. Then the Fourier series of $f$ converges to $f$ pointwise.

  $ lim_(n -> infinity) 1/2 a_0 + sum_(k=0)^n
    underbrace(a_k cos(k x) + b_k sin(k x), "n"^"th" "order partial sum")
   = f(x) $

  where

  $ a_k = 1 / pi integral_(-pi)^pi f(x) cos(k x) $
  $ b_k = 1 / pi integral_(-pi)^pi f(x) sin(k x) $

  Furthermore, we can say something about the error:

  $ |f(x) - S_n (x)| <= 1 / sqrt(pi n) sqrt(integral_(-pi)^pi |f'(t)|^2 d t) $

  *Remark*: not all functions satisfy these "nice" conditions (in fact, most "real life" functions don't).
]

== Reflections

Is $P_n / S_n$ a "good" approximation? What does "good" even mean? Is $P_n / S_n$ the "best" approximation?

Clearly, $P_n$ isn't the best. Here's a counterexample:

#example[counterexample][
  $f: [-1, 1] -> R$ given by
  $ f(x) = cases(
    e^(-1/x^2) space "if" space x in [-1, 0) union (0, 1),
    0 space "if" space x = 0
  ) $

  The problem here is that the derivatives at 0 are always going to be 0. This also happens with the other term $f^((k)) (x) = q_k (1/x) e^(-1/x^2)$ which is $0$ at $x = 0$, so it's differentiable.

  With that we would conclude that the approximation is just the 0 function, but clearly we can do better than the 0 function for approximating the exponential.

  This is an example of a #link("https://en.wikipedia.org/wiki/Flat_function")[_flat function_].
]

The approximations in the previous examples are "good" (or "good enough") in the sense of the _supremum norm_, $||f - P_n (x)||_infinity$.

#definition[
  Let $S subset RR$. Then the _supremum_ of $S$ (if it exists) is the smallest upper bound for elements in $S$.

  #example[
    Given $S = { 1 - 1/n : n in NN }$, we have $S = {0, 1/2, 3/4, 4/5, ...}$ so the supremum is $1$ (even though $1 in.not S$)
  ]

  We use a shorthand notation:

  $ sup { f(x) : x in I } = sup_(x in I) f(x) $

  The _infimum_ can be defined very similarly, but being the biggest lower bound instead.
]

#definition[
  The _supremum norm_ is the following:

  $ ||f - g||_infinity = sup_(x in I) |f(x) - g(x)| <= sup $
]

Our concepts of "good" depends on our concsers of "size" or "distance".

= Concept of "good" and "distance".

In the first example, the definition of "good" was that the "distance" between $f(x)$ and $P_n (x)$ is at most $10^(-6)$, where "distance" was the supreme norm.

== Metric spaces

#definition[
  A _metric space_ is a pair $(X, d)$ where $X$ is a set and $d$ is a _metric_, i.e., a distance function $d: X times X -> [0, infinity)$ that satisfies:
  + $d(x, x) = 0$ (reflexivity)
  + $d(x, y) > 0$ if $x != y$ (positivity)
  + $d(x, y) = d(y, x)$ (symmetry)
  + $d(x, y) + d(y, z) >= d(x, z)$ (triangle inequality)
]

#example[classic ones][
  - $X = RR$ and $d(x, y) = |x - y|$
  - $X = CC$ and $d(z, w) = |z - w|$
  - $X = RR^n$ and $d(x, y) = ||x - y|| = sqrt(sum_(k=1)^n (x_k - y_k)^2)$
]

#example[
  $X = C[a, b] = { f : [a, b] -> RR and f "is continuous"}$

  ($C[a, b]$ is a continuous function between $a$ and $b$)

  $d(f, g)$ can be:
  $ d(f, g) = ||f - g||_([a, b]) = sup_(x in [a, b]) |f(x) - g(x)| $
  $ d(f, g) = integral_a^b |f(x) - g(x)| d x $
  $ d(f, g) = integral_a^b |f(x) - g(x)|^2 d x $
]


#definition[
  A sequence $(x_n)_(n=0)^infinity$ (or just $x_n$) _converges_ to $x_infinity$ iff $lim_(n -> infinity) d(x_n, x_infinity) = 0$.

  In words, a sequence _converges_ if there is a value where values eventually get arbitrarily close to.

  We use the notation $x_n -> x_infinity "as" n -> infinity$ or $lim_(n->infinity) x_n = x_infinity$
]

#example[
  $X = C([-1, 1])$ and $d(f, g) = ||f - g||_infinity$

  $f(x) = e^x$ and $P_n (x) = sum_(k = 0)^n 1/k! x^k$. Then, we know that $||f - P_n||_infinity $ converges to $0$.
]

#definition[
  Given a metric space $(X, d)$, a subset $X subset X$ is called _sequentially compact_ if every sequence $x_n$ in $K$ has a subsequence that converges to a point $x_infinity in K$.
]

The idea is that a compact space has no "punctures" or "missing endpoints", i.e., it includes all limiting values of points #link("https://en.wikipedia.org/wiki/Compact_space")[(wikipedia)].

#theorem("Bolzano-Weierstrass")[
  If $K$ is closed and bounded then $K$ is compact.
]

#example[
  Take $X = CC$ and $d(z, w) = |z - w|$. Then the closed unit disc, $K = { z in CC : |z| <= 1 }$ is compact after Bolzano-Weierstrass.

  Meanwhile, the non-closed unit disc is not compact. Take for instance $x_n = 1 - 1/n$. Every subsequence in this sequence converges to one, but in the non-closed disc $1$ is not included in the set.
]

#example[
  $ X = { p : p "is a polynomial with real coefficients" } $
  $ d(p, q) = integral_(-1)^1 |p(x) - q(x)| d x $

  $ K = { p in X : deg p <= n } $
  Where the coefficients satisfy:

  $ forall k = 0, .., n : |a_k| <= 1 $

  And to clarify: $p(x) = sum_(k = 0) ^n a_k x^k$

  *Question*: is this _compact_?

  Let $p_j$ be any sequence in $K$.

  $p_j (x) = sum_(k=0)^n a_(j, k) x^k$

  Let's check $n = 0$. Then, $p_j (x) = a_(j,0) x^0 = a_(j,0)$. We see $p_j (x)$ is just the real numbers which are known compact. In other words, $(a_(j,0))^*$ converges.

  When checking $n = 1$ we have $p_j (x) = a_(j,0) + a_(j,1) x$
  There exists a subsequence $(p_j)^*$ such that $a_(j,0)^*$ converges).

  There exists a subsequence $((p_j)^*)^*$ such that $a_(j,1)^*$ converges).

  There exists a subsequence $(((p_j)^*)^*)^*$ such that $a_(j,2)^*$ converges).

  Therefore there exists a subsequence $tilde(p)_j$ such that $tilde(a)_(j,k) -> a_(infinity,k) in [-1, 1] forall k in NN$ (where $tilde(p)_j$ represents a number of subsequence operators nesting). #todo[Why?]

  We still have to check #todo[what do we have to check?]

  We can do this using the triangle inequality:

  $ d(tilde(p)_j, p_infinity) = integral_(-1)^1 |tilde(p)_j (x) - p_infinity (x)| d x \
    <= integral_(-1)^1 sum_(k=0)^n | tilde(a)_(j,k) - a_(infinity,k) |x|^k d x \
    <= integral_(-1)^1 sum_(k=0)^n |tilde(a)_(j,k) - a_(infinity,k)|  d x \
    <= 2 sum_(k=0)^n |tilde(a)_(j,k) - a_(infinity,k)| \
  $
]

