#import "../format.typ": *

#show: notes[Computational Techniques for Differential Equations]

#todo[Content from first lecture]

= Finite Difference (FD)

There are three ways to do this.
- Interpolation
- Taylor expansion
- Monomials

*Objective*: Given a function $u$ and their derivatives, want to find

$ (d^((k)) u) / (d x^((u))) approx omega_1 u(x_1) + omega_2 u(x_2) + ... + omega_n u(x_n) $ where $n >= k + 1$, with ${ (x_1, u(x_1)), (x_2, u(x_2)), ..., (x_n, u(x_n))}$ are known

Most usual ways to do this

$ D_+ u(x_0) = (u(x_0 + h) - u(x_0))/h = u'(x_0) + h/2 u''(x_0) + O(h^2) $
$ D_- u(x_0) = (u(x_0 - h) - u(x_0))/h = u'(x_0) - h/2 u''(x_0) + O(h^2) $
$ D_0 u(x_0) = (u(x_0 + h) - u(x_0 - h)) / (2h) $

$D_0$ works better than $D_plus.minus$ because we essentially use three points.

The residual of $D_+$ is $D_+ u(x_0) = underbrace(u'(x_0) + h/2 u''(x_0) + O(h^2), R(x_0) space "(residual)")$

$ omega_1 = 1/h, omega_2 = -1/h $

== Approaches
=== Taylor expansions

Let's take three points we know, such as ${ (x_0, u(x_0)), (x_1, u(x_1)), (x_2, u(x_2))}$ (this is forwards, you can do it also centered and backwards).

The equation we want is of the following form:

$ u'(x_0) + R(x_0) = omega_0 u(x_0) + omega_1 u(x_0 + h) + omega_2 u(x_0 + 2h) $

We can Taylor expand the points at $x_0 + h$ and $x_0 + 2h$.

$ u(x_0 + h) = u(x_0) + h u'(x_0) + h^2/2 u''(x_0) + h^3/6 u'''(x_0) + O(h^4) $
$
  u(x_0 + 2h) = u(x_0) + 2h u'(x_0) + 4h^2/2 u''(x_0) + 4h^3/3 u'''(x_0) + O(h^4)
$

Where $h$ is positive and small.

We substitute the two last equations into the first one, to get:

$
  u'(x_0) + R(x_0) &= (omega_0 + omega_1 + omega_2) u(x_0) + (omega_1 + 2 omega_2) h u'(x_0) \
  &+ 1/2 (omega_1 + 4 omega_2) h^2 u''(x_0) + 1/6 (omega_1 + 8 omega_2) h^3 u'''(x_0) \
  & + O(h^4)
$

We want to minimize the residual, basicaully make $u'(x_0) approx u'(x_0) + R(x_0)$ so $R(x_0) approx 0$.

Therefore, we can take the necessary coefficients to match them to the original function.

Namely, $ cases(
  (omega_0 + omega_1 + omega_2) = 0,
  (omega_1 + 2 omega_2) h = 1,
  1/2 (omega_1 + 4 omega_2) h^2 = 0,
) $

The solution is

$ omega_0 = -3/(2h), omega_2 = 2/h, omega_3 = -1/(2h) $

Finally, we can substitute them in


$u'(x_0) = -3/(2h) u(x_0) + 2/h u(x_0 + h) - 1/(2h) u(x_0 + 2h) + underbrace(1/3 h^2 u''(x_0) + O(h^3), -R(x_0))$

=== Polynomial interpolation

If we want to approximate the $k^"th"$ derivative, we need to have $k + 1$ points.

For three points ${ x_0, x_0 + h, x_0 + 2h }$, the equation is the following:

$
  u(x) + R(x) = P_2(x) = \
  underbrace(((x-(x_0 + h)) (x - (x_0 + 2h))) / (h(2h)), l_0(x)) u(x_0) + \
  underbrace(((x - x_0) (x - (x_0 + 2h))) / (h(-h)), l_1(x)) u(x_0 + h) + \
  underbrace(((x - x_0) (x - (x_0 + h))) / (2h(h)), l_2(x)) u(x_0 + 2h)
$

#todo[I did $l_2$ by hand, it might be wrong]

This is the _Lagrange_ form.

Working it out you get the same result as before.

=== Monomials

This works by enforcing the equation $x'(x_0) approx omega_0 u(x_0) + omega_1 u(x_0 + h) + omega_2 u(x_0 + 2h)$ to be exact for monomials.

Namely, those monomials are ${ 1, x, x^2, ..., x^(n-1) }$ for $n$ points. In practice, we can use ${ 1, (x - x_0), (x - x_0)^2, ...}$ to get more $0$s involved, since the spans of the two sets are the same, just with a translated basis.

Here, we replace $u(x)$ in the rhs by
$
                                         u(x_0) & eq.triple 1 \
  =>
  omega_0 dot 1 + omega_1 dot 2 + omega_2 dot 1 & = (1)'|_(x=x_0) = 0 \
             => omega_0 + omega_1 + omega_2 = 0
$

$
  u(x_0) eq.triple (x - x_0) => \ omega_0 (x_0 - x_0) + omega_1 (x_0 + h - x_0) + omega_2 (x_0 + 2h - x_0) = (x - x_0)'|_(x_0 = x) = 1
$

$
  u(x) eq.triple (x - x_0)^2
  \ => omega_0 (x_0 - x_0)^2 + omega_1 (x_0 + h - x_0)^2 + omega_2 (x_0 + 2h - x_0)^2 = ((x - x_0)^2)'|_(x=x_0)
  \ => omega_1 h^2 + omega_2 4h^2 = 0
  \ => #todo[?]
$

== Approximations error

Let's look at it with $D_+$. The approximation error is the residual, $R(x) = D_+ u(x_0) - u'(x_0) = 1/2 h u''(x_0) + 1/6 h^2 u'''(x_0) + O(h^3)$. We can omit everything except the first term, since that overshadows the rest, so we take $1/2 h u''(x_0)$.

We define the error as $E(h) = 1/2 h u''(x_0)$, with small positive $h$ ($0 < h << 1$). Let's see how the error propagates.

#figure(table(
  columns: (auto, auto),
  $ h $, $ E(h) $,
  $ h_0 $, $ E(h_0) $,
  $ h_0/2 $, $ E(h_0 / 2) = 1/4 h_0 u''(x_0) = 1/2 E(h_0) $,
  $ h_0/4 $, $ E(h_0 / 4) = 1/8 h_0 u''(x_0) = 1/4 E(h_0) $,
  $ colon.tri.op $, $ colon.tri.op $,
))

This might look kind of obvious, but this is the key to numerics. This is the way to check if a numerical method works. It needs to converge at the expected rate mathematically, otherwise your code is probably wrong.

Now, let's look at the ratio of the errors. Here, $E(h_0) / E(h_0 / 2) = 2$. Generally, the error has the shape $E(h) = A h^P$ (so here $A = 1/2 u''(x_0)$, $p = 1$).

If we take the log of the error we get $log(E(h)) = log A + p log h$. We call $p$ the _convergence order_.

We know that in our case $p = 1$, but we can also take a look at the error ratio, and we should find that $log_2(E(h_0) / E(h_0/2)) = 2$ because $log_2(E(h_0)) / E(h_0/2) = p$ in general.

To be proper, the formula is this:

$ log |E(h)| approx log|C| + p log h $

We have an $approx$ because $E(h)$ is not _actually_ $A h^p$, that's just the leading term, but the approximation gets better as $h$ becomes smaller.

We can check the formula by plotting $log |E(h)|$ and $log h$ and checking that the slope is $p$. You could also do a fitting to see what the best $p$ is, what is the deviation, and so on and so forth.

*NOTE*: In practice, there are machine precision errors with floating point which are proportional to $E_m prop 1/h$.

#example[Exercise 2][
  We have the operator:

  $
    D_3 u(overline(x)) = 1/(6h) (
      2u(overline(x) + h) + 3u(overline(x)) - 6u(overline(x) - h) + u(overline(x) - 2h)
    )
  $

  For 2)

  This is to approximate $u'(x)$ so $D_3 u(x) = u'(x) + O(h^3)$.

  We do Taylor expansion:

  $ D_3 u(x) - u'(x) = h^3/12 u^(("iv"))(x) + O(h^4) $

  For 3)

  We want essentially $ u''(x_0) approx omega_1 u(x_0 - h) + omega_2 u(x_0) + omega_3 u(x_0 + h) $

  We can do
  1. Taylor expansions

  We've seen this.

  2. Lagrange interpolation

  What degree do we need? Answer: 2, because we have 3 points.

  $
    u(x) approx P_2(x) = u(x_0 - h) l_0(x) + u(x_0) l_1(x) + u(x_0 + h) l_2(x) \
    u''(x)|_(x_0) approx P''_2(x)|_(x_0) = u''(x_0 - h)|_(x_0) l_0(x) + u''(x_0)|_(x_0) l_1(x) + u''(x_0 + h)|_(x_0) l_2(x)
  $

  Evaluating the derivatives at grid values is important for solving PDEs, because the first step is to discretize the grid.

  The solutions are $1/h^2, -2/h^2, 1/h^2$.

  3. Monomials

  #todo[Do at home]

  You should get

  $
    cases(
      omega_1 + omega_2 + omega_3 = 0,
      omega_1 - omega_3 = 0,
      omega_1 + omega_3 = 2/h^2,
    )
  $

  For 4)

  Th exercise is about finding the approximation of a fourth derivative with a few points. But let's first illustrate an example by trying to approximate the second derivate.

  #todo[this is very messy, I should do the second derivative and the fourth separately. The second with 5 points is a shit ton of work for the same result as 3 points]

  $
    u''(x) approx \
    omega_0 ( u(x_0) - 2h u'(x_0) + (2h)^2/2 u''(x_0) - (2h)3/6 u'''(x_0) + (2h)^4/24 u^("iv") + ... ) \
    + omega_1 ... \
    + omega_2 ... \
    + omega_3 ... \
    + omega_4 ... \
    = \
    (omega_0 + omega_1 + omega_2 + omega_3 + omega_4) u(x_0) + \
    (-2 omega_0 - omega_1 + omega_2 + omega_3 + 2omega_4) u'(x_0) h + \
    (2 omega_0 + 1/2 omega_1 + 0 omega_2 + 1/2 omega_3 + 2omega_4) h^2 u'(x_0) + \
    (-4/3 omega_0 - omega_1/6 + omega_3/6 + 4/3 omega_4) h^3 u'''(x_0) + \
    (2/3 omega_0 + 1/24 omega_1 + 1/24 omega_3 + 2/3 omega_4) h^4 u^(("iv"))(x_0) + \
    underbrace(O(h^5), O(h) "but really" O(h^2) "because" h^5 "term cancels out (probably)")
  $

  The final term is $O(h^5)$ so the error is $O(h)$ all terms are proportional to $1/h^4$ #todo[?], and actually it's going to be $O(h^2)$ because the $h^5$ term will cancel out.

  We could do more #todo[in the original, or here or are we just lazy??], but imagine we stop here.

  We add the constraints:

  $
    cases(
      omega_0 + omega_1 + omega_2 + omega_3 + omega_4 = 0,
      -2 omega_0 - omega_1 + omega_2 + omega_3 + 2omega_4 = 0,
      2 omega_0 + 1/2 omega_1 + 0 omega_2 + 1/2 omega_3 + 2omega_4 = 1/h^2,
      -4/3 omega_0 - omega_1/6 + omega_3/6 + 4/3 omega_4,
      2/3 omega_0 + 1/24 omega_1 + 1/24 omega_3 + 2/3 omega_4,
    )
  $

  #todo[Exercise: Finish writing this to get $u''(x_0) approx D_5 u(x_0) + O(h^4)$]

  The thing is that here we have two free variables. These are not necesarilly optimal because we have an $h^3$ error still, so we have the same error as $D_+ D_+ u(x_0)$, while using $5$ points. The point is that you have to use all points and find the highest polynomial degree.a

  Fundamentally, I guess you don't get any benefit by trying to approximate a second derivative with higher polynomials
]


== Finite difference methods for 1D boundary condition values problems

Differential equations

$ u''(x) = f(x), x in (0, 1) $

Boundary conditions:

$ cases(u(0) = alpha, u(1) = beta) $

#[

  #let m = 10
  #let x = range(m)
  Steps:

  1. #[
      Define a grid:
      We start by defining a grid. In 1D, for $m$ points inside the bound, we take $(m + 2)$  points for evenly spaced points of distance $h = 1/(m + 1)$. The points are $x_j = j h$ for $j = 0, 1, 2, ..., m + 1$.

      // #lq.diagram(lq.plot())
    ]

  2. #[
      Define "grid function", which are the numerical values that we use to approximate our exact function.

      ${ U_0, U_1, U_2, ..., U_(m+1) }$ such that $u(x_j) = U_j$.

      #lq.diagram(lq.plot(x, x))
    ]

  3. #[
      Discretization:

      We have $u''(x_j) = f(x_j)$ for $j = 1, 2, ..., m$ that we want to approximate (but the formula as written is exact).

      For example we could do this with the 3 point finite difference formula $D u(x_j) = 1/h^2 (u(x_(j - 1)) - 2 u(x_j) + u(x_(j + 1))) = u''(x_0) + O(h^2)$.
    ]
]

$u$ is unknown exactly. We have an approximation $u(x_j) = U_j$. This implies that $1/h^2 (U_(j-1) - 2U_j + U_(j+1)) = f(x_j)$ for all grid points $= 1, ..., m$. This is the _finite difference discretization_.

Now we apply the boundary conditions. In general we have $1/h^2 (U_(j-1) - 2U_j + U_(j+1)) = f(x_j)$ but for $x_1$ and $x_(m+1)$ we have different conditions because we know the boundaries:

$
  f(x_1) = 1/h^2 (U_0 - 2U_1 + U_2) \
  f(x_1) = 1/h^2 (alpha - 2U_1 + U_2) \
  1/h^2 (-2U_1 + U_2) = f(x_1) - alpha/h^2
$


$
  f(x_m) = 1/h^2 (U_(m-1) - 2U_m) + U_(m+1)) \
  f(x_m) = 1/h^2 (U_(m-1) - 2U_m) + beta) \
  1/h^2 (U_(m-1) - 2U_m) = f(x_m) - beta/h^2
$

#faint[Up to here we don't need a computer, and, in numerics, the longer you delay using the computer it is better.]


Now, we have a system of $m$ equations with $m$ unknowns.

We can pose it as a matrix:

$
  A = 1/h^2 mat(
    -2, 1, 0, 0, dots.down;
    1, 0, 0, dots.down, 0;
    0, 0, dots.down, 0, 0;
    0, dots.down, 0, 0, 1;
    dots.down, 0, 0, 1, -2
  )
$

$
  arrow(u) = mat(
    u_0 = alpha;
    u_1;
    u_2;
    u_3;
    dots.v;
    u_(m-1);
    u_m = beta;
  )
$

$
  "rhs" = mat(
    f(x_1) - alpha/h^2;
    f(x_2);
    f(x_3);
    dots.v;
    f(x_(m - 1));
    f(x_m) - beta/h^2;
  )
$

Where $A arrow(u) = "rhs"$. We solve for $arrow(u)$ to get the values.

== Error analysis

What is the error $E_j = U_j - u(x_j)$?

For the example of $u'' = f(x)$ for $x in [0, 1)$ and $u(0) = alpha$, $u(1) = beta$.

Let's take the _local truncation error_:

$
  tau_j = 1/h^2 underbrace((u(x_(j-1)) -2u(x_j) + u(x_(j+1))), "FD-2") - underbrace(f(x_j), u''(x_j))
$

And if we take the Taylor series:

$
  tau_j & = u''(x_j) + 1/12 h^2 u^(("iv"))(x_j) + O(h^4) - f(x_j) \
        & = 1/12 h^2 u^(("iv"))(x_j) + O(h^4)
$

How to go from this to the global error $E_j$?

Let's take the exact vector

$
  arrow(hat(U)) = mat(u(x_1); dots.v; u(x_n))
$

which implies that $arrow(tau) = A arrow(hat(U)) - arrow(F) => A arrow(hat(U)) = arrow(F) + arrow(tau)$

We have from before $A arrow(U) = arrow(F)$. If we subtract each other we have:

$
  A underbrace((U - arrow(hat(U))), E) = -arrow(tau)
$

From we can extract that the error $E_j = U_j - u(x_j)$ is $E = U - arrow(U)$. This can also be written as:

$ A arrow(E) = -arrow(tau) $

The error vectors and the local truncation errors are related in this way. So, in the example $1/h^2 (E_(j-1) - 2E_j + E_(j + 1)) = -tau_j$. Another way to put it is that $e'' = -tau$.

At the boundary, for $j=1$, $1/h^2(E_0 - 2E_1 + E_2)$, where $E_0$ is $0$ because we enforce the boundary conditions. That is, at the boundaries:

$ E_0 = E_(m+1) = 0 $

Then, to answer the original question, for a given $h$ we have $A^h E^h = -tau^h$. If $A$ is invertible, we have that $E^h = -(A^h)^(-1) tau^h$.

#todo[what dis $arrow.b$]
$
  norm(E^h) = norm((A^h)^(-1) tau^h) <= norm((A^h)^(-1)) dot norm(tau^h) overparen(=, ?)C h^2
$

This $C h^2$ is just what we get numerically in the specific example, that the error decreases quadratically with $h$. We should be able to show this numerically, so it's what we need to show.

We have that $norm(tau^h) = O(h^2)$ through the Taylor series. We would need to have that $norm(A^h)^(-1) = O(1)$ to make the formula above coincide.

Let's try to see that $norm(A^h)^(-1) = O(1)$. But first, let's define a few things:

#definition[Stability][
  Given a numerical scheme $A^h U^h = F^h$, it is _stable_ if for any $h < h_0 << 1$, the matrix $A^h$ is invertible and the norm of the inverse is bounded by some constant:

  $ norm((A^h)^(-1)) <= C $
]

#definition[Consistency][
  Given a numerical scheme $A^h U^h = F^h$, it is _consistent_ if  the local truncation error goes to $0$ as $h$ goes to $0$, i.e.:

  $ norm(tau^h) -> 0 space "as" space h -> 0 $
]

#definition[Convergence][
  Given a numerical scheme $A^h U^h = F^h$, it is _convergent_ if the error goes to $0$ as $h$ goes to zero, i.e.:

  $ norm(E^h) -> 0 space "as" space h -> 0 $
]

#theorem[
  Consistency and stability implies convergence.

  That is, a scheme that is consistent and stable is also necessarily convergent.
]


Let's try to find if $norm((A^h)^(-1)) = O(1)$. This is true if the scheme is _stable_. To find if it is stable we are going to use the _edge_ norm, where $norm(A)_2 = rho(A) = max_(1<=p<=m)|lambda_p|$ where $lambda_p$ are the eigenvalues.

We want to compute $norm(A^(-1))_2 = rho(A^(-1)) = max |lambda_p^(-1)| = (min|lambda_p|)^(-1)$.

We take the eigengrid function: $u_j^p = sin(p pi j h)$ for $j = 1,...,m$. This works because

$
  A u_j^p & = 1/h^2 (u^p_(j-1) u^p_j u^p_(j+1)) \
          & = lambda_p u^p_j \
          & = underbrace(2/h^2(cos(p pi h) - 1), lambda_p) u^p_j
$

#todo[Homework: check this $arrow.t$]

Since we have the eigenvalues, we can see that the smallest absolute eigenvalue is $lambda_1 = 2/h^2 (cos(pi h) - 1)$ which we can Taylor expand to

$ lambda_1 = 2/h^2 (-1/2 pi^2 h^2 + O(h^4)) = -pi^2 + O(h^2) $

So $norm(A^(-1))_2 = 1/pi^2$ as $h -> 0$. This is the result we were expecting.
