#import "../format.typ": *

#show: notes(
  subtitle: [Approximating functions by "simpler" other functions of some specified type.],
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

  $
    lim_(n -> infinity) 1/2 a_0 + sum_(k=0)^n
    underbrace(a_k cos(k x) + b_k sin(k x), "n"^"th" "order partial sum")
    = f(x)
  $

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
  $
    f(x) = cases(
      e^(-1/x^2) space "if" space x in [-1, 0) union (0, 1),
      0 space "if" space x = 0
    )
  $

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

= Metric spaces

In the first example, the definition of "good" was that the "distance" between $f(x)$ and $P_n (x)$ is at most $10^(-6)$, where "distance" was the supreme norm.

== Definition and compactness

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

  $f(x) = e^x$ and $P_n (x) = sum_(k = 0)^n 1/k! x^k$. Then, we know that $||f - P_n||_infinity$ converges to $0$.
]

#definition[
  Given a metric space $(X, d)$, a subset $K subset X$ is called _sequentially compact_ if every sequence $x_n$ in $K$ has a subsequence that converges to a point $x_infinity in K$.
]

The idea is that a compact space has no "punctures" or "missing endpoints", i.e., it includes all limiting values of points #link("https://en.wikipedia.org/wiki/Compact_space")[(wikipedia)].

#definition[
  Given a metric space $(X, d)$ a subset $F subset X$ is _closed_ if every convergent sequence has a limit in $F$.

  A set $O subset X$ is _open_ if it is the complement of a closed set.

  *Remark*: These are generalizations of the same concepts in $RR^n$.
]

#definition[
  Given a metric space $(X, d)$ and two subsets $A, B subset X$, we say that _$A$ is dense in $B$_ if for every $p in B$ and every $epsilon > 0$ there exists an element $q$ in $A$ such that $d(p, q) < epsilon$.

  In other words, any element in $B$ can be approximated arbitrarily close by any element in $A$.

  #example[The rational numbers are dense in the reals (i.e., $QQ$ is dense in $RR$) since you can approximate to arbitrary precision any real number by rational numbers.]
]

#theorem("Bolzano-Weierstrass")[
  If $K subset RR^n$ is closed and bounded then $K$ is compact.
]

#example[
  Take $X = CC$ and $d(z, w) = |z - w|$. Then the closed unit disc, $K = { z in CC : |z| <= 1 }$ is compact after Bolzano-Weierstrass.

  Meanwhile, the non-closed unit disc is not compact. Take for instance $x_n = 1 - 1/n$. Every subsequence in this sequence converges to one, but in the non-closed disc $1$ is not included in the set.
]

#example[
  $ X = { p : p "is a polynomial with real coefficients" } $
  $ d(p, q) = integral_(-1)^1 |p(x) - q(x)| d x $

  $ K = { p in X : deg p <= n } $

  Where the coefficients satisfy: $|a_k| <= 1$

  And to clarify: $p(x) = sum_(k = 0)^n a_k x^k$

  *Question*: is this _compact_?

  Let $p_j$ be any sequence in $K$.

  $p_j (x) = sum_(k=0)^n a_(j, k) x^k$

  Let's check $n = 0$. Then, $p_j (x) = a_(j,0) x^0 = a_(j,0)$. We see $p_j (x)$ is just real numbers below one (bounded) which are known compact. In other words, $(a_(j,0))^*$ converges.

  When checking $n = 1$ we have $p_j (x) = a_(j,0) + a_(j,1) x$. We know we can take a subsequence $(p_j)^*$ such that the term $a_(j,0)$ converges, so we only have to care about the term $a_(j,1) x$. But $a_(j,1)x$ is isomorphic to the real numbers below one (as before), so you can take a subsequence $((p_j)^*)^*$ such that $a_(j,1)$ converges.

  Doing the same thing, there exists a subsequence $(((p_j)^*)^*)^*$ such that $a_(j,2)^*$ converges), etc.

  Therefore there exists a subsequence $tilde(p)_j$ such that $tilde(a)_(j,k) -> a_(infinity,k) in [-1, 1] forall k in NN$ (where $tilde(p)_j$ represents a number of subsequence operators nesting).

  We still have to check that this makes $p$ itself converges.

  $
    d(tilde(p)_j, p_infinity) = integral_(-1)^1 |tilde(p)_j (x) - p_infinity (x)| d x
    \ <= integral_(-1)^1 sum_(k=0)^n | tilde(a)_(j,k) - a_(infinity,k)| |x|^k d x
    \ <= integral_(-1)^1 sum_(k=0)^n |tilde(a)_(j,k) - a_(infinity,k)| d x
    \ <= 2 sum_(k=0)^n |tilde(a)_(j,k) - a_(infinity,k)|
    \ = 0
  $

  Indeed, it converges.
]

#theorem[Existence of the "best approximation"][
  Given a metric space $(X, d)$ and a compact subspace $K subset X$, then there exists a point in $K$ of minimal distance to any other $X$.
] <theorem-existence-of-best-approximation>

#proof[
  Let $delta = inf { d(p, q) : q in K }$ (the largest lower bound for distance to a point $q$ in $K$). Since $d$ is positive-definite the infimum exists because it can be $0$. It is a general property of infimums that there exists a sequence $q_n$ in $K$ such that $lim_(n->infinity) d(p, q_n) = delta$ #faint[(because, handwavey-ly, the infimum is either on the set where the convergent sequence is obvious or it is right on the edge of the set where you can get arbitrarily close to the infimum)].

  We can suspect that $q_infinity$ is this "closest point" to $p$ in $K$. But we need to prove that $q_n$ converges and that this convergence point is also in $K$.

  We start by remembering that $K$ is compact so we can take a subsequence $(q_n)^*$ of $q_n$ such that $q_n^*$ converges to a point $q_infinity^*$ as $n -> infinity$. Since the original sequence converges, the subsequence has to converge to the same point, so $q_infinity^* = q_infinity$. But, by the definition of compactness $q_infinity^*$ is in $K$ so $q_infinity$ is also in $K$.

  The question now is whether $d(p, q_infinity)$ is minimal. We know that $d(p, q_infinity) >= inf { d(p, q) : q in K } = delta$, from the definition of the infimum.

  We can also invoke the triangle inequality, where for any $n$, $d(p, q_infinity) <= d(p, q_n^*) + d(q_n^*, q_infinity)$. If we take the limit as $n -> infinity$ we get

  $
    d(p, q_infinity) &<= underbrace(d(p, q_infinity^*), delta) + underbrace(d(q_infinity^*, q_infinity), 0) = delta
  $

  So $d(p, q_infinity) <= delta$ and $d(p, q_infinity) >= delta$ so $d(p, q_infinity) = delta$. $p_infinity$ exists and is a best approximation.
]

*Remark*: Not all sets of approximation functions are compact.

== More structured spaces (normed and inner product space)

Let's consider examples of approximating functions:
- Taylor:  Algebraic polynomials
- Fourier: Trigonometric polynomial

These two live in spaces that have extra structures, which can be useful to spell out. Polynomials are in vector (or linear) space, you can add and scale them together. But also all the distance functions $d$ come from a _norm_.

#definition[
  A _normed linear space_ $(X, ||dot||)$ where $X$ is a linear spce and $||dot||$ is a _norm_.

  A _norm_ is a function $||dot|| : X -> [0, infinity)$ such that:
  + $||f||$ > 0 unless $f = 0$ (positive definite)
  + $||lambda f|| = ||lambda|| ||f||$ for $lambda in RR$ (or $CC$) (homogeneity)
  + $||f + g|| <= ||f|| + ||g||$ (triangle inequality)
]

The following definitions are less important.

#definition[
  A _Banach_ space is a normed linear space that also satisfies that a Cauchy-sequence $f_n$ (where $lim_(m,n -> infinity) ||f_n - f_m|| = 0$), then there exists a point $g in X$ such that $lim_(n -> infinity) ||f_n - g||$
]

#definition[
  A _Hilbert_ space is a _Banach_ space where we have
  $ ||f + g||^2 + ||f - g||^2 = 2||f||^2 + 2||g||^2 $
]

#example[
  $X = C[a, b]$ and $||f|| = integral_a^b |f(x)| d x$ is a Banach space.
]

#definition[
  Given a normed linear space $(X, ||dot||)$ then a subset $B subset X$ is called _bounded_ if there exists an $R > 0$ such that $B subset { f in X : ||f|| <= R}$.

  This is sometimes called a _closed ball with radius R_.
]

#theorem[
  In a normed linear space $(X, ||dot||)$, every closed, bounded, finite-dimensional $F subset X$ is _compact_.
] <theorem-closed-bounded-finite-compact>

We can prove this but we need more definitions.

#definition[Continuity][
  Given two metrix spaces $(X, d)$ and $(Y, tilde(d))$, and a mapping $phi: X -> Y$ then $phi$ is _continuous at $x in X$_ iff for any sequence $x_n subset X$ which converges to $x$ we have also that $phi(x_n) -> phi(x)$

  $phi$ is _continuous_ if it is continuous at every point $x in X$.
]

#theorem[
  Any continuous map $phi : X -> Y$ sends compact sets to compact sets.

  If $K subset X$ is compact, then $phi(K) = { phi(x) : x in K }$ is compact.
] <continuous-sends-compact-compact>

#proof[of @theorem-closed-bounded-finite-compact][
  Let $F$ satisfy the theorem conditions (closed, bounded, finite-dimensional).

  For $F$ to be finite-dimensional it means that there are some linearly independent vectors $q_n$ the span of which $F$ is a subset (i.e., for every $f in F$ there exist some coefficients $lambda_n$ such that $f = sum lambda_j q_j$)

  Now consider the map $phi : RR^n -> X$ given by $phi(lambda) = phi(lambda_1, lambda_2, ..., lambda_n) = l_1 g_1 + ... + l_n g_n$. We use the norm on $RR^n$ as $||lambda||_m = max |l_j|$.

  Claim: $phi$ is continuous. Let's take a sequence $(lambda)_k$ that converges to $lambda$. We need to check that $phi((lambda)_k)$ converges to $phi(lambda)$, so in turn we need to show that $||phi((lambda)_k) - phi(lambda))||$ goes to $0$ as $k -> infinity$.

  $
    ||phi((lambda)_k) - phi(lambda)|| &= ||sum_j (lambda_j)_k q_j sum lambda_j g_j||
    \ &= ||sum_j (lambda_j)_k - lambda_j) q_j||
    \ &<= sum_j ||((lambda_j)_k - lambda_j) g_j||
    \ &<= sum_j ||((lambda_j)_k - lambda_j)|| ||g_j||
    \ &<= sum_j ||((lambda_j)_k - lambda_j)|| ||g_j||
    \ &<= ||(lambda_j)_k - lambda_j|| n R " where " ||g_j|| <= R forall j=1,..n
    \ &= 0 dot n R
    \ &= 0
  $

  So $phi$ is continuous.

  Let $M = { lambda in RR^n : phi(lambda) in F }$. $phi$ sends $M$ to $F$ ($F = phi(M)$) so, by @continuous-sends-compact-compact, if we show that $M$ is compact, $F$ is also compact.

  Claim: $M$ is compact. #todo[prove it]
]

#definition[
  An _inner-product space_ is a tuple $(X, innerproduct(dot, dot))$ with an inner product which is a function $innerproduct(dot, dot): X times X -> RR$ (or $CC$) such that:
  + $innerproduct(f, f) > 0$ unless $f = 0$ (positive-definite)
  + $innerproduct(f, g) = innerproduct(g, f)$ (or $innerproduct(f, g) = overline(innerproduct(g, f))$) (symmetry)
  + $innerproduct(alpha f + beta g, h) = alpha innerproduct(f, h) + beta innerproduct(g, h)$ for $alpha, beta in RR "or" CC$ (linearity)

  #example[
    - $X = RR^n$ and $innerproduct(x, y) = x_1 y_1 + ... + x_n y_n$
    - $X = CC^n$ and $innerproduct(z, w) = z_1 overline(w)_1 + ... + z_n overline(w)_n$
    - $X = C[a,b]$ and $innerproduct(f, g) = integral_a^b f(x) overline(g(x)) d x$
  ]

  *Remark*: Any inner-product spaces defines a normed linear space through $||f|| = sqrt(innerproduct(f, f))$
]

#theorem[
  Given an inner-product space $(X, innerproduct(dot, dot))$ and two elements $f,g in X$ with a norm defined as $||f|| = sqrt(innerproduct(f, f))$:

  + $|innerproduct(f, g)| <= ||f|| ||g||$ (Cauchy-Schwarz)
  + $||f + g|| <= ||f|| + ||g||$ (triangle inequality)
  + $||f + g||^2 + ||f - g||^2 = 2||f||^2 + 2||g||^2$ (parallelogram law)
  + If $||innerproduct(f, g)|| = 0$ then $||f + g||^2 = ||f||^2 + ||g||^2$ (Pythagoras)
]

#definition[
  If $innerproduct(f, g) = 0$ we say that $f$ is orthogonal to $g$.

  If $G = { g_1, ..., g_n }$ is a set such that $innerproduct(g_i, g_j) = 0$ if $i != j$ then $G$ is an _orthogonal_ set. If $innerproduct(g_i, g_j) = delta_(i,j)$ then $G$ is _orthonormal_.
]

#theorem[Gram-Schmidt][
  Any set of linearly independent vectors $G = { g_1, ..., g_n }$ can be used to construct an orthonormal set.
]

#theorem[Best approximation in inner-product space][
  Given an inner-product space $(X, innerproduct(dot, dot))$ and an element $f in X$. Let $G = {g_1, ..., g_n}$ be an orthonormal set in $X$. Then $||sum_k c_k g_n - f||$, will be minimal exactly when $c_k = innerproduct(f, g_k)$.

  *Remark*: This is a very strong result.
]

#proof[
  Define $c_k = innerproduct(f, g_k)$ and we have $g = sum_k c_k g_k$.

  We want to show that $||g - f|| < ||h - f||$ for any $h in X without { g }$.

  Notice that $g - f$ is orthogonal to any $g_k$:

  $
    innerproduct(f - g, g_k) & = innerproduct(f, g_k) - innerproduct(g, g_k) \
                             & = c_k - innerproduct(sum_j c_j g_j, g_k) \
                             & = c_k - sum_j innerproduct(c_j g_j, g_k) \
                             & = c_k - sum_j c_j innerproduct(g_j, g_k) \
                             & = c_k - c_k \
                             & = 0
  $

  $
    innerproduct(f - g, h) = 0 quad forall h = a_1 g-1 + ... + a_n g_n in "span"(g_1, ..., g_n)
  $

  Remember the Pythagorean theorem.

  $
     ||h - f||^2 & = ||(h - g) + (g - f)||^2 \
                 & = ||h - g||^2 + ||g - f||^2 \
                 & >= ||g - f||^2 \
    => ||h - f|| & >= ||g - f||
  $

  This proves that $g$ is _some_ minimum but we want to prove it's the _only_ minimum.

  We can check where $g$ is unique by realizing that equality happens only when $||h - g|| = 0$ by the definition of the norm.

  Therefore $g$ is the unique best approximation.
]


We would like to have more general approximations, with other kinds of spaces with less restrictions. To do this we need to introduce more definitions:

#definition[Convexity][
  $K$ is _convex_ if for all $f,g in K$ we have $theta f + (1-theta)g in K$.

  $theta f + (1-theta)g$ is a line parametrization of $g -> f$ so a set is convex if the line between any two points in a set is also entirely inside the set.

  An equivalent definition is that $k$ is convex if for all $f_1, ..., f_n$ we have

  $ sum_(i=1)^n theta_i f_i in K $

  when $theta_i >= 0$ such that $sum_(i=1)^n theta_i = 1$.

  The above is called a _convex linear combination_.
]

#definition[Convex hull][
  The _convex hull_ of a set $A subset X$ of a linear space $X$ is defined as:

  $
    cal(H)(A) = { g = sum_(i=1)^m theta_i f_i : m in NN, theta_i >= 0, sum_(i=1)^m theta_i = 1, f_1,...,f_m in A }
  $

  In words, $cal(H)$ is the set of all convex linear combinations of elements in $A$.
]

#theorem[The convex hull of any set is convex.]

#theorem[Carathéodory][
  For an $n$-dimensional linear space $X$, and a subset $A subset X$, then every element $f in cal(H)$  can be written as $f = sum_(i=0)^n theta_i f_i$ (where $theta_i$ and $f_i$ satisfy the previous conditions).
]

#proof[
  Let $g in cal(H)$. $g = sum_(i=0)^k theta_i f_i$ where $f_0,...,f_k in A, theta_i >= 0, sum_(i=0)^n theta_i = 1$. Assume $k$ is minimal with this property, so that $theta_i > 0$.

  Claim: We can do this with $k <= n$.

  Assume, $k > n$.

  $
    sum_(i=0)^k theta_i g_i = sum_(i=0)^k (theta_i f_i - theta_i g) = g - (sum_(i=0)^k) = 0
  $

  We have a set ${ g_1, ..., g_k }$ which has at least $n+1$ elements since they have $k+1$ elements and $k>=n$. These are, then, linearly dependent since the dimension of the space is $n$.

  Therefore, there exist constants $alpha_i$ such that $alpha_1 g_1 + ... + alpha_k g_k = 0$.

  Defining $alpha_0 = 0$, for any $lambda in RR$ we can consider the sum:

  $
    sum_(i=0)^k (theta_i - lambda alpha_i) g_i = underbrace(sum_(i=0)^k theta_i g_i, 0) - underbrace(lambda sum_(i=0)^k alpha_i g_i, 0) = 0
  $

  We can cleverly choose $lambda$ to get results. Namely, we can choose $lambda = alpha_i/theta_i$ for the $i$ that corresponds with the minimal $theta_i$, which makes $theta_i - lambda alpha_i = 0$ and keeps all other $theta_i - lambda alpha_i$ nonnegative. The combination is still non-trivial since $theta_0 - lambda alpha_0 = theta_0 > 0$.

  We can use this to create a new linear combination that produces $g$. Given that $g_i = f_i - g$, we have:

  $
    sum_(i=0)^k (theta_i - lambda alpha_i) g_i = sum_(i=0)^k (theta_i - lambda alpha_i)(f_i - g) = sum_(i=0)^k (theta_i - lambda alpha_i)f_i - g sum_(i=0)^k (theta_i - lambda alpha_i)
  $

  We can divide by the sum that multiplies $g$, since it's positive, so:

  $
    g & = 1/(sum_(i=0)^k (theta_i - lambda alpha_i)) sum_(i=0)^k (theta_i - lambda alpha_i) f_i
    \ & = sum_(i=0)^k (
      (theta_i - lambda alpha_i) / (sum_(j=0)^k (theta_j - lambda alpha_j)) f_i
    )
  $

  This is a convex combination. The sum of the coefficients is:

  $
    sum_(i=0)^k ( (theta_i - lambda alpha_i) / (sum_(j=0)^k (theta_j - lambda alpha_j))) = 1
  $

  We know also that all coefficients are nonnegative.

  The thing is that at least one of the coefficients $theta_i - lambda alpha_i$ is $0$, so $k$ cannot be minimal if $k > n$! Therefore, for $k$ to be minimal we need that $k <= n$.
]

#corollary[
  For a linear space $X$ of dimension $n$ and a compact subset $A subset X$. Then $cal(H)(A)$ is compact.
]

#proof[
  Take any sequence $(f_k)_k subset cal(H)(A)$. Then,

  $f_k = sum_(i=0)^n theta_(k i) f_(k i)$ where $theta_(k i) >= 0$, $sum_(k=0)^n theta_(k i)$ and $f_i in A$.

  Since $A$ is compact, there is a convergent subsequence of the $(f_k)^*$ such that $(f_(k i))_k$ converges to some limit $f_i$ as $k -> oo$.

  Once having taken this limit, we can also do this for $theta_i$ since they're contained in $[0, 1]$ which is clearly compact. So, $(f_k)^(**)$ such that $theta_(k i) -> theta_i$.

  The converging limit should be

  $ f = sum_(i=0)^n theta_i f_i $

  We have to show that this is convex. This is simple since $f_i in A$, $theta_i >= 0$ because all $theta_(k i)$ are positive, and $lim_(k->oo) sum_(i=0)^n theta_(k i) = 1 = sum_(i=0)^n lim_(k->oo) theta_(k i) = 1$. So $f$ is indeed convex and thus is in the convex hull.

  Therefore, every sequence has a subsquence that converges to an element of a set, so the convex hull $cal(H)$ is compact.
]

#theorem[Existence][
  Let $(X, norm(dot))$ be a normed linear space. Given a compact subset $A subset X$ which is finite dimensional then there exists a $g in A$ which is "a" best approximation to $f$, such that $norm(f - g) = inf_(h in A) norm(f - h)$
]

#proof[
  Let $f_0 in A$. Consider the set $S = { h in A : norm(h - f) <= norm(f_0 - f)}$.

  We want to use @theorem-existence-of-best-approximation. We need to prove, then, that $S$ is compact. Using @theorem-closed-bounded-finite-compact, proving that $S$ is closed, bounded and finite-dimensional, then $S$ is compact.

  - Finite-dimensional: Trivial, given in premise. #sym.checkmark
  - Bounded: $S$ is bounded since $norm(h - f) <= norm(f_0 - f)$ #sym.checkmark
  - Closed: Let $(f_k)_k$ be a convergent sequence in $S$ with $f_k -> tilde(f)$ as $k -> oo$.
    $norm(tilde(f) - f) = norm(lim_(k->oo) f_k - f) = lim_(k->oo) norm(f_k - f) <= lim_(k->oo) norm(f_0 - f) = norm(f_0 - f)$ so $tilde(f) in S$.

    Therefore, $S$ is compact by @theorem-closed-bounded-finite-compact.
]

#example[
  Let $X$ be the space of sequences $(x_n)_n subset RR$ that converge to $0$. This becomes a normed linear space with the following norm:

  $ norm((x_n)) = max_n |x_n| $

  Let $A$ be the subspace of sequences $(x_n)$ such that $sum_(k=1)^oo 2^(-k) x_n = 0$.

  Pick any $(y_n)_n in.not A$. Then $sum_(k=1)^oo 2^(-k) y_k = lambda != 0$ and in fact, without loss of generality, $lambda > 0$.

  #underline[Claim]: the minimal distance of $A$ to $(y_n)_n$ is at most $lambda$.

  Consider sequences:
  - $x_((1)) = -2/1 (lambda, 0, 0, ...) + y$
  - $x_((2)) = -4/3 (lambda, lambda, 0, ...) + y$
  - $x_((n)) = -2^n/(2^n-1) (underbrace(lambda\, lambda\, ...\, lambda, n), 0, ...) + y$
  Where $y = (y_1, y_2, y_3, ...)$

  Then the norm $norm(x_((n)) - y)$ is:

  $
    norm(x_((n)) - y) & = -2^n/2^(n-1) (underbrace(lambda\, lambda\, ...\, lambda, n), 0, ...) + y
    \ & = 2^n/(2^n-1) lambda -> lambda "as" n -> oo
  $

  $
    sum_(k=1)^oo 2^(-k) x_((n), k)
    & = sum_(k=1)^oo - (2^n 2^(-k))/(2^n - 1) + sum_(k=1)^oo 2^(-k) y_k \
    & = -(2^n)/(2^n - 1)(1 - 2^(-n)) lambda = -lambda + lambda = 0
  $

  Suppose, to reach a contradiction, that there exists $(z_n)_n in A$ such that $norm((z_n) - (y_n)) <= lambda$. Consider $n$ such that $|z_k - y_k| <= 1/2 lambda$. This can happen because all sequences in $A$ converge to $0$. Then,

  $
    lambda = sum_(k=1)^oo 2^(-k) y_k
    & = sum_(k=1)^n 2^(-k)(y_k - z_k) \
    & <=^(triangle "ineq.") sum_(k=1)^oo 2^(-k) |y_k - z_k| \
    & = sum_(k=1)^(n-1) 2^(-k) |y_k - z_k| + sum_(k=n)^oo 2^(-k) |y_k - z_k| \
    & <= sum_(k=1)^(n-1) 2^(-k) norm((y_k) - (z_k)) + sum_(k=n)^oo 2^(-k) 1/2 lambda \
  $

  Now we use the asssumption that $norm((z_n) - (y_n)) <= lambda$,

  $
    & <= lambda(sum_(k=1)^(n-1) 2^(-k)) + 1/2lambda(sum_(k=n)^oo 2^(-k)) \
    & = (1 - 2^(-n)) lambda < lambda \
  $

  and we reach a contradiction!

  Therefore, for any $(z_n) in A$ we have $norm((z_n) - (y_n)) > lambda$. We can come arbitrarily close to $lambda$ with elements of $A$ but we can never reach it.
]

#definition[Uniform convexity][
  A linear space $(X, norm(dot))$ is _uniformly convex_ if for every $epsilon > 0$ there exists a $delta$ such that $norm(f - g) <= epsilon$ if $norm(f) = norm(g) = 1$ and $norm(1/2 (f + g)) < 1 - delta$
]

#theorem[Existence and unicity][
  Given a uniformly convex Banach space $(X, norm(dot))$ and a closed and convex subset $A subset X$ and an element $f in X$, there exists a unique element $g in A$ that minimizes $norm(f - g)$
]

#definition[Strictly convex][
  A normed linear space $(X, norm(dot))$ is _strictly convex_ if for some $f,g in X$ where $norm(f) = norm(g) = norm(1/2(f+g)) = 1$, then $f = g$.
]

#theorem[Existnce and unicity][
  Given a normed linear space $(X, norm(dot))$ that is strictly convex, then for an element $f in X$ and a finite-dimensional subspace $A subset X$ there exists a unique element $g in A$ that minimizes the $norm(f - g)$.
]

Those are all the theorems we'll see related to convexity.

= How to approximate functions in $norm(dot)_oo$

Given a continuous function $f : [a, b] -> RR$ ($f in C[a, b]$), can we make $norm(f - p)_oo = sup_(x in [a,b])|f(x) - p(x)|$ arbitrarily small with polynomials $p$?

Let's try one approach:

Given points $x_0, x_1, ..., x_n in [a,b]$ we consider a polynomial of degree no greater than $n$, such that the polynomial agrees with the values of $f$ in these points, that is, $p_n(x_n) = x_n$, where $p_n(x) = sum_(i=0)^n c_n x^n$.

#theorem[
  It's not "a" polynomial, it's "the" polynomial, since there is only one such polynomial. In fact, it's called the Lagrange interpolation polynomial.
]

There is a proof where you can write it as a linear system where the matrix is a Vandermonde matrix which are known to be invertible.

Doing the interpolation explicitly:

$
  p_n(x) = sum_(i=0)^n f(x_i) product_(j=0, j != i)^n (x - x_j) / (x_i - x_j) = sum_(i=0)^n f(x_i) W(x) / ((x - x_i) W'(x_i))
$

Where $W$ is the Wilkinson polynomial

$ W(x) = W_n(x_0, x_1, ..., x_n, x) = product_(i=0)^n (x - x_i) $

And $lim_(x -> x_i) W(x) / ((x - x_i) W'(x_i)) = lim_(x -> x_i) (W(x) - W(x_i))/(x - x_i) 1/(W'(x_i)) = W'(x_i) 1/(W'(x_i)) = 1$.

So this rewriting works.

Question now is how well does the interpolating polynomail approximates $f$? (in $norm(dot)_oo$).

#theorem[
  $norm(f - P_n)_oo = sup_([a,b])|f(x) - P_n (x)| <= 1/((n+1)!) norm(f^((n+1)))_oo norm(W)_oo$
]

So, if you know that the norm of all derivatives are bounded by $(n+1)!$ (and $norm(W)_oo$)

#proof[
  We will prove that for every $x$ in $[a, b]$ there exists a $xi in (a, b)$ such that $f(x) - P_n(x) = 1/((n+1)!) f^((n+1)) (xi) W(x)$. The points $x_n$ are given by both sides being zero.

  For the rest of the points, $W(x_n)$ is not zero, so the quantity to prove is:

  $ f(x) - P_n (x) = 1/((n+1)!) f^((n+1))(xi) $

  We are going to define $phi(y) = f(y) - P_n(y) - lambda W(y)$, where $lambda = lambda_x = (f(x) - P_n (x))/W(x)$. This is useful because $phi(y)$ has roots at $x_n$ and one more! Namely, $phi(x) = f(x) - P_n (x) - (f(x) - P_n (x))/W(x) W(x)$. So it has roots on $x, x_0, x_1, ..., x_n$, therefore at $n + 2$ points.

  We need to quickly involve another theorem:

  #theorem[Rolle][
    Suppose $phi : [c,d] -> RR$ is continuous on $[c,d]$, differentiable on $(c,d)$ and $phi(c) = phi(d)$, then there exists a $xi in (c, d)$ such that $phi'(xi) = 0$
  ]

  In our case, by Rolle's theorem, there exist $n + 1$ points where $phi'(y) = 0$. We can repeat this argument: there exist $n$ points where $phi''(y) = 0$, there exist $n - 1$ points where $phi'''(y) = 0$, etc. In general, there exists $n + 2 - j$ points where the $(dif^j)/(dif y^n) phi(y) = 0$. At the $n+1$th derivative, there exists at least one point where $phi^((n+1))(xi) = 0$, which means that:

  $
    phi^((n+1))(x) & = f^((n+1))(xi) - P_n^((n+1))(xi) - lambda W^((n+1))(xi) \
                   & = f^((n+1))(xi) - 0 - lambda (n+1)! \
  $

  And put the value of $lambda$ in...

  $
    f^((n+1))(xi) = (f(x) - P_n (x))/W(x) (n+1)! => 1/(n+1)! f^((n+1)) (xi) W(x)
  $

  This clearly implies that

  $ norm(f(x) - P_n (x))_oo <= 1/(n+1)! norm(f^((n+1)))_oo norm(W)_oo. $
]

Turns out while every selection of points can get arbitrarily close, there is some optimal choice of points ${x_0, x_1, ..., x_n}$ which makes $norm(W)_oo$ as small as possible.

#theorem[
  $ norm(W)_oo = sup_(x in [a,b])|product_(i=0)^n (x-x_i)| $

  is minimal when

  $ x_i = -(b+a)/(b-a) + 2/(b-a)cos(pi (2i+1)/(2n)) $

  for $i = 0, ..., n$.
]

#proof[
  First we should translat the problem to the range $[-1, 1]$. So, consider instead $tilde(W)(x) = (2/(b - a))^(n+1) W((b-a)/(2x) + (b+a)/2)$ where $tilde(W) : [-1, 1] -> RR$.

  $
    tilde(W)(x) & = (2/(b - a))^(n+1) W((b-a)/(2x) + (b+a)/2) \
    & = (2/(b-a))^(n+1) product_(i=0)^n ((b-a)/2 x + (b+a)/2 - x_i) \
    & = product_(i=0)^n (x - underbrace((b+a)/(b-a) + 2/(b-a) x_i, y_i))
  $

  Claim: $y_i = cos(pi (2i + 1)/(2(n+1)))$.

  We know that $cos((n + 1)theta) = sum_(k=0)^(n+1) a_k (cos(theta))^k$. The polynomial $T_(n+1)(x) = sum_(k=0)^(n+1) a_k x^k$ is called the Chebyshev polynomial. It satisfies the following:

  $ T_(n+1) (cos(theta)) = cos((n+1)theta) $

  Clearly, the minima and maxima are at $(n+1)theta = pi k$ for $k in ZZ$, and the zeros are at the halfpoints $(n+1)theta = 1/2 pi + pi i$ for $i in ZZ$. The zeroes of the Chebychev polynomials can be obtained from the $theta$s of the zeroes of $cos((n+1)theta)$. So, $theta = pi (2i + 1)/(2(n+1))$. Here we only need $i=0,...,n$because of the cyclicity of $cos$. So the zeroes of $T_(n+1)$ are at $cos(pi (2i+1)/(2(n+1)))$.

  The norm $norm(T_(n+1))$ is, then, $1$.

  We will know show that any polynomial of the shape $product_(i=0)^n (x - x_i)$ has norm no less than one.

  Suppose that there exists a $tilde(W)$ such that $norm(hat(W))_oo < norm(tilde(W))_oo$.

  For $k=0$, the difference between $hat(W)$

  $tilde(W)$ and $hat(W)$ alternate between positive and negative, and by the intermediate value theorem there are $n+1$ points where $tilde(W) - hat(W) cos(theta) = 0$. We know that $tilde(W)$ and $hat(W)$ have leading term $1 dot x^(n+1)$so they cancel out and $tilde(W) - hat(W)$ have degree $n$ and $n+1$ zeroes, and this is only possible if $tilde(W) - hat(W) = 0$, so that $tilde(W) = hat(W)$. Therefore $tilde(W)$ is optimal.
]

Question: does $P_n$ indeed converge to $f$ in supremum norm? Actually, it doesn't! For instance, for $f(x) = 1/(1+x^2)$.

#theorem[Weierstrass][
  Given a continuous function $f in C[a,b]$ then, for an arbitrary $epsilon > 0$, there exists a polynomial $P$ such that $norm(f - P)_oo < epsilon$.
]

The proof is instructive, since it gives you the procedure itself to get decent approximations of $f$ by polynomials. It uses Brnstein polynomials.

#proof[

  We define a map $B_n : C[a,b] -> C[a,b]$ (or $RR_n [x]$ polynomails of degree $<= n$ with real coefficients). This map is also called an operator. We are going to also work in $[a, b] = [0, 1]$, and you can always go back to $[a, b]$ by rescaling, so this is with no loss of generality.

  The operator $B_n$ is:

  $ (B_n f)(x) = sum_(k=0)^n f(k/n) binom(n, k) x^k (1-x)^(n-k) $

  The claim is that
  $ lim_(n->oo) norm(f - B_n f)_oo = 0. $

  We will invoke the following theorem:
  #theorem[
    Suppose $L_n$ is a monotone linear operator on $C[0, 1]$. Then, $lim_(n->oo) norm(L_n f - f)_oo = 0$ for all $f in C[0, 1]$ iff $lim_(n->oo) norm(L_n f - f) = 0$ for $f(x) = 1$, $f(x) = x$ and $f(x) = x^2$.

    Linear means what you think it means and a monotone operator means that $(L_n f) (x) >= (L_n g) (x)$ when $f(x) >= g(x)$.
  ]

  $B_n$ is a monotone linear operator. This we also assume to know.

  Now we need to show that the theorem condition holds for $1, x$ and $x^2$.

  - $f(x) = 1$, then $(B_n f)(x) = sum_(k=0)^n f(k/n) binom(n, k) x^k (1-x)^(n-k)$.

    We can  use Newton's binomial formula, so $(B_n f)(x) = 1 = f(x)$, so $norm(f(x) - B_n f(x))_oo = norm(0)_oo = 0$.

  - $f(x) = x$, then $(B_n f)(x) = sum_(k=0)^n k/n binom(n, k) x^k (1-x)^(n-k)$

    We can start the sum at $k=1$, so
    $
      & = sum_(k=0)^n k/n n!/(k!(n-k)!) x^k (1-x)^(n-k) \
      & = sum_(k=0)^n (n-1)!/((k-1)!(n-k)!) x^k (1-x)^(n-k) \
      & = sum_(k=1)^(n-1) (n-1)!/(k!(n-1-k)!) x^(k-1) (1-x)^(n-1-k) \
      & = x (x+1 - x)^(n-1) = f(x)
    $

  - Finally, $f(x) = x^2$. Instead of doing it directly, we are going to do $B_n (f - 1/n g)$ for $g(x) = x$. This is now:

    $
      (B_n (x^2 - x/n))(x) & = sum_(k=0)^n ( (k/n)^2 - 1/n k/n ) binom(n, k) x^k (1-x)^(n-k) \
      & = sum_(k=0)^n ( (k(k-1))/n^2 ) n!/(k! (n-k)!) x^k (1-x)^(n-k) \
      & = n(n+1)/n^2 sum_(k=2)^n ( (k(k-1))/n^2 ) (n-2)!/((k-2)! (n-k)!) x^k (1-x)^(n-k) \
      & = (1-1/n) f(x)
    $

    Then, $sup|B_n f(x) - f(x)| = sup|(1-1/n)f(x) + 1/n x - f(x)| = sup|-1/n f(x) + 1/n x| <= sup|-1/n x^2| + sup|1/n x| = -1/n sup|x^2| + 1/n sup|x|$, which clearly goes to $0$ as $n -> oo$. By the theorem above, the limit

    $ lim_(n->oo) norm(B_n f - f)_oo = 0 $
]

= Measure theory

#let scr(it) = text(features: ("ss01",), box($cal(it)$))
#definition[Topology][
  A family $tau subset scr(P)(X)$ is a _topology_ on $X$ if it satisfies the following three conditions:
  + $emptyset, X in tau$.
  + $tau$ is closed under arbitrary unions.
  + $tau$ is closed under finite intersections.
]

#example[Topologies][
  A couple of topologies:

  $
    tau_0 = { emptyset, X } \
    tau_F = P(x)
  $
]

#definition[Metric space][
  A set $(X, d)$ is a _metric space_ if $d$ is a distance, that is,
  + $d(x,y) >= 0$
  + $d(x, y) = d(y, x)$
  + $d(x, y) + d(y, z) >= d(x, z)$

  for all $x, y, z$ in $X$.
]

#definition[Open set][
  Given a metric space $(X, d)$ and an $epsilon > 0$ we can define

  $ B_d(x, epsilon) = { phi in X : d(x, y) < epsilon } subset X $

  We say that $A subset X$ is an _open set_ if for any $a in A$ there exists $r_a > 0$ such that $B_d(a, r_a) subset A$ for all $a in A$.
]

The collection of all closed sets in a topologial space also has interesting properties.

If $(X, tau)$ is a topological space, then the family $Omega subset scr(P)(X)$ given by $Omega = { E in scr(P)(X) : E^c = X \\ E in tau }$ satisfies:

+ $emptyset, X in Omega$.
+ $Omega$ is closed under finite unions.
+ $Omega$ is closed under arbitrary intersections.

This is the dual form of the definition of a topology.

#definition[$sigma$-Algebra][
  Let $X$ be a set and let $scr(F)$ be a collection of subsets of $X$. Then $scr(F)$ _is a $sigma$-algebra of susbets of $X$_ (or "$sigma$-algebra on $X$" for short) if it satisfies:

  + $emptyset in scr(F)$.
  + for all $A in scr(F)$, $A^c$ is also in $scr(F)$.
  + $scr(F)$ is closed under countable unions.
]

#theorem[basic properties of $sigma$-algebras][
  - First two conditions imply that $emptyset^c = X$ and $X in scr(F)$.
  - First and third conditions imply that $scr(F)$ is closed under finite unions.
  - Last two conditions imply that $scr(F)$ is closed under countable intersections.
]

#definition[Measurable set][
  Elements of a $sigma$-algebra are called _measurable sets_.
]

#definition[Measurable space][
  An ordered pair $(X, scr(F))$, where $X$ is a set and $scr(F)$ is a $sigma$-algebra on $X$, is called a _measurable space_.
]

*Remark*: Most topologies are not $sigma$-algebras, and most $sigma$-algebras are not topologies. However, the smallest and largest topologies and $sigma$-algebras on a set are the same.

- Indiscrete topology on $X$, $tau_0 = { emptyset, X }$ is a $sigma$-algebra.
- Discrete topology on $X$, $tau = scr(P)(X)$ is also a $sigma$-algebra.

#definition[Simple $sigma$-algebra generated by $A$][
  For a set $X$, if $A subset X$ and $A != emptyset$, then we call the collection $scr(F)_A = { emptyset, A, A^c, X }$ a _simple $sigma$-algebra on $X$ generated by $A$_
]

#theorem[
  For completeness, a simple $sigma$-algebra generated by a subset $X$ is a $sigma$-algebra.
]

#example[
  Take $X = { x_1, x_2, ..., x_n }$. The collection $scr(F)_{x_1} = { emptyset, {x_1}, { x_2, ..., x_n}, X }$ is a set generated by ${ x_1 }$.
]

A few more examples and claims:
- The collection $scr(F) = { A subset X : A "or" A^c "is countable" }$ is a $sigma$-algebra on $X$ (distinct from $scr(P)(X)$ iff $X$ is uncountable).

#definition[The $sigma$-algebra generated by an arbitrary family][

  Let $scr(A)$ be an arbitrary family of subsets of $X$. Then, the $sigma$-algebra generated by $scr(A)$ is the smallest linear algebra which contains every set in $scr(A)$.

  #faint[Note that $scr(A)$ itself is not necessarily a $sigma$-algebra!]

  This algebra is denoted by $sigma(scr(A))$
]

#theorem[$sigma$-agebras for subspaces][
  For a subset $A subset X$:

  + If $(X, scr(F))$ is a measurable space, the collection ${ A inter B : B in scr(F) }$ is a $sigma$-algebra on $A$.
  + If $(A, Sigma)$ is a measurable space, the collection $B subset X : B inter A in Sigma$ is a $sigma$-algebra on $X$.
]

#theorem[The $sigma$-algebra generated by a function][
  If $f : X -> Y$ and $Sigma$ is a $sigma$-algebra on $Y$, then the collection

  $ sigma(f) := { f^(-1)(B) : B in Sigma } $

  is a $sigma$-algebra on $X$.
]

Fiven a collection $scr(C)$ of subsets of a set $X$, we already know how to find the smallest (weakest) possible topologu $tau$ on $X$ sich that $scr(C) subset tau$. This is a relatively concrete procedure.

How do we do the same thing for $sigma$-algebras?

#theorem[
  Let $X$ be a set, let $I$ be a non-empty indexing set, and let ${ scr(F)_i : i in I }$ be $sigma$-algebras on $X$. Then,


  $ inter.big_(i in X) scr(F)_i $

  is also a $sigma$-algebra on $X$.
]

#proof[(sketch)][
  Let

  #let f = $scr(F)_*$;
  $ scr(F)_* = inter.big_(i in X) scr(F)_i $

  + Since $emptyset in scr(F)_i$ for all $i$, then $emptyset in f$.
  + For a set $A$ to be in $f$ then it is in all $scr(F)_i$. Since $scr(F)_i$ are $sigma$-algebras then $A^c$ is also in every $scr(F)_i$, so since it is in every $scr(F)_i$ it is in $f$.
  + #todo[fill this in]
]


#definition[Borel sets][
  Let $(X, tau)$ be a topological space. We set $scr(BB) = sigma(tau)$, the $sigma$-algebra on $X$ generated by the collection of open sets in $tau$.

  We call the sets in $scr(B)$ the _Borel sets_ (or the Borel measurable subsets of $X$).
]

Although these definitions seems clean and simple, it is far from obvious which subsets of $RR$ are or are not Borel sets. For instance, consider $tau$ the usual topology on $RR$:

$ tau subset scr(B) = sigma(tau) $

So, every open set in $RR$ is a Broel set. Taking complements, closed sets are also Borel sets. In particular, single-point sets ${ a }$ are Borel sets.

Countable unions of closed sets are Borel sets. Countable union of closed sets are called $F_sigma$ sets. In particular, all countable subsets of $RR$ are Borel sets, e..g., $QQ$ is a Borel set. Then so is $QQ^c = RR \\ QQ$. Similarly, the complements of $F_sigma$ sets, called $G_delta$ sets, are Borel sets.

#definition[Measure and measure space][
  Let $(X, scr(F))$ be a measurable space. A _measure_ on $scr(F)$ is a function $mu : scr(F) -> [0, oo)$, satisfying:
  + $mu(emptyset) = 0$
  + whenever $A_1, A_2, ...$ are pairwise disjoint sets in $scr(F)$ then

  $ mu (union.big_(n=1)^oo A_n ) = sum_(n=1)^oo m(A_n) $

  We call the triple $(X, scr(F), mu)$ is a _measure space_.
]

The most important example of a measure for us will be the Lebesgue measure on the Borel sets (or the Lebesgue measurable sets) in $RR$. Lebesgue measure gives the "total length" of these sets.

For now we can look at simpler sets.

#example[
  - Counting measure: just count the number of elements. This is a measure.
  - The characteristic function $xi_A : X -> { 0, 1 }$ for some subset $X subset A$ where it is $1$ if $x in A$ or $0$ if $x in.not A$
]

#definition[Complete measures][
  A measure space $(X, scr(F), mu)$ is _complete_ and $mu$ is a _complete measure_ if:

  $ forall N subset B in scr(F): m(B) = 0 => N in scr(F) "and" mu(N) = 0. $

  That is, for all subsets $N subset B$
]

#example[
  - The counting measure is complete.
  - Given $x_0 in X$. The Dirac-$delta$ measure at $x_0$ ($1$ if $x_0 in A$, $0$ otherwise) is a complete measure.

  Both of these are because $scr(F) = P(X)$.
]

#theorem[
  On a measure space $(X, scr(F), mu)$ we consider:

  $
    overline(scr(F)) = { E subset X : exists A,B in scr(F) "with" A subset E subset B "and" mu(B \\ A) = 0 }.
  $

  and foe $E in overline(scr(F))$ we define $overline(mu)(E) = mu(A)$. Then:
  + $overline(scr(F))$ is a $sigma$-algebra, the smallest such that $scr(F) subset overline(scr(F))$.
  + $overline(mu)$ is a complete measure, extending $mu$ #faint[where "extending $mu$" means that if we restrict $overline(mu)$ to $scr(F)$ then we get $mu$].
]

#definition[Outer measure][
  An _outer measure_ is a set function $mu^* : scr(P) -> [0, oo)$ where:
  + $mu^*(emptyset) = 0$.
  + If $A subset B$ then $mu^*(A) <= mu^*(B)$. #faint[This is basically monotony.]
  + If ${ A_j }_(j in NN) subset X$  then $mu^* ( U_(j=1)^oo A_j) <= sum_(j=1)^oo mu^* (A_j)$.
]

Now we want to find  collection $scr(M) in scr(P)(X)$ such that $mu^*|_scr(M)$ is a measure.

#definition[$mu^*$-measurable][
  Given an outer measure $mu^*$ on $X$, a subset $M subset X$ is _$mu^*$-measurable_ if for every subset $A subset X$ then

  $ mu^*(A) = mu^*(A inter M) + mu^*(A \\ M) $
]

#theorem[Carathéodory theorem][
  Consider $scr(M) = { M subset X : M "is" mu^*"-measurable" }$, then:
  + $scr(M)$ is a $sigma$-algebra.
  + $mu = mu^*|_scr(M)$ is a complete measure.
]

#faint[Apparently Carathéodory was Nazi...]

The point is that it is easier to construct outer measures and via this theorem we can construct regular measures.


#definition[
  collection $scr(E) subset scr(P)(X)$ is a semi-algebra if:
  + $emptyset in scr(E)$
  + $E, F in scr(E) => E union F in scr(E)$, i.e., closed under union.
  + $E in scr(E) => E^c = union.big_(i=1)^n F_i$ for $F_i in scr(E)$ and all $F_i$ are disjoint. I.e., the complement can be written as a finite union of elements of $scr(F)$

  They are called "semi" because they resemble of semi-open intervals.
]

#definition[
  A set function $mu$ is $sigma$-finite on a set $X$ if:

  $ X = union.big_(j=1)^oo X_j "with" mu(X_j) < oo "for all" j. $
]

#theorem[Carathéodory-Hopf extension theorem][
  Consider a semi-algebra $scr(E) subset scr(P)(X)$ and $mu_0 : scr(E) -> [0, oo)$ a countably additive function. We define for all $A in scr(P):$

  $
    mu^*(A) = inf { sum_(j=1)^oo mu_0 (E_j) : E_j in scr(E), A subset union.big_(j=1)^oo E_j }
  $

  then:

  + $mu^*$ is an outer measure and $mu = mu^*|_scr(M)$ is a complete measure which is an extension of $mu_0$. That is, $mu^*(E) = mu_0(E)$ for all $E in scr(E)$.
  + If $mu_0$ is finite, then
    + $mu$ is the unique measure which is an extension of $mu_0$ to $sigma(scr(E))$
    + $scr(M) = overline(sigma(scr(E)))$. That is, $scr(M)$ is the copmletion relative to $mu$ of $sigma(scr(E))$. #faint[the overline is just to indicate that $scr(M)$ is complete.]
]

#definition[
  A semi-open interval in $RR^n$ is a set of the type $I = I_1 times ... times I_n$ where each $I_j$ is a semi-open interval in $RR$.
]

#theorem[
  there exists a unique measure space $(RR^n, scr(M), m)$ such that $scr(M) = overline(scr(B))(RR^n)$ and $m|_scr(E) = mu_0$. In particular:
  + For all $M in scr(M)$, $M = B union N$ where $B in scr(B)(RR^n)$ and $m(N) = 0$.
  + For all $N in scr(M)$ with $m(N) = 0$ there exists $A in scr(B)(RR^n)$ with $N subset A$ and $mu(A) = 0$

  This is called the _Lebesgue_ measure in $RR^n$.
]

#definition[Measurable function][
  Given a measurable space $(X, scr(F))$ and a topological space $(Y, scr(T))$, a mapping $f : X -> Y$ is mesurable if $f^(-1)(V) in scr(F)$ for all $V in scr(T)$.
]

#example[
  If $(X, scr(F))$ is a measurable space and $A in scr(F)$ then $xi_A : A -> RR$  is a measurable function, and if $B in.not scr(F)$ then $chi_B$ is not a measurable function. In particular, $chi_QQ$ is mesurable
]

// #example[
//   Consider $X = {1, 2, 3}$ and the $sigma$-algebra $scr(A) = { emptyset, X }$. The function

//   $ f : X -> RR space f(1) $
// ]
