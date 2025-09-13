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


]

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
] <compact-normed-linear-spaces>

We can prove this but we need more definitions.

#definition[Continuity][
  Given two metrix spaces $(X, d)$ and $(Y, tilde(d))$, and a mapping $phi: X -> Y$ then $phi$ is _continuous at $x in X$_ iff for any sequence $x_n subset X$ which converges to $x$ we have also that $phi(x_n) -> phi(x)$

  $phi$ is _continuous_ if it is continuous at every point $x in X$.
]

#theorem[
  Any continuous map $phi : X -> Y$ sends compact sets to compact sets.

  If $K subset X$ is compact, then $phi(K) = { phi(x) : x in K }$ is compact.
] <continuous-sends-compact-compact>

#proof[of @compact-normed-linear-spaces][
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

#let innerproduct(x, y) = $lr(angle.l #x, #y angle.r)$
#definition[
  An _inner-product space_ is a tuple $(X, innerproduct(dot, dot))$ is a linear space $X$ with an inner product which is a function $innerproduct(dot, dot): X times X -> RR$ (or $CC$) such that:
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
  Given an inner-product space $(X, innerproduct(dot, dot))$ and two elements $f,g in X$ with a normed defined as $||f|| = sqrt(innerproduct(f, f))$:

  + $|innerproduct(f, g)| <= ||f|| ||g||$ (Cauchy-Schwarz)
  + $||f + g|| <= ||f|| + ||g||$ (triangle inequality)
  + $||f + g||^2 + ||f - g||^2 = 2||f||^2 + 2||g||^2$ (parallelogram law)
  + If $||innerproduct(f, g)|| = 0$ then $||f + g||^2 = ||f||^2 + ||g||^2$ (Pythagoras)
]

#definition[
  If $innerproduct(f, g) = 0$ we say that $f$ is orthogonal to $g$.

  If $G = { g_1, ..., g_n }$ is a set such that $innerproduct(g_i, g_j) = 0$ if $i != j$ then $G$ is an _orthogonal_ set. If $innerproduct(g_i, g_j) = delta_(i,j)$ then $G$ is _orthonormal_.
]

#theorem[Gramm-Schmidt][
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
