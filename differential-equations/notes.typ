#import "../format.typ": *

#show: notes[Computational Techniques for Differential Equations]

#todo[Content from first lecture]

= Finite Difference (FD)

There are three ways to do this.
- Interpolation
- Taylor expansion
- Monomials

*Objective*: Given a function $u$ and their derivatives, want to find

$ (d^((k)) u) / (d x ^((u))) approx omega_1 u(x_1) + omega_2 u(x_2) + ... + omega_n u(x_n) $ where $n >= k + 1$, with ${ (x_1, u(x_1)), (x_2, u(x_2)), ...,  (x_n, u(x_n))}$ are known

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
$ u(x_0 + 2h) = u(x_0) + 2h u'(x_0) + 4h^2/2 u''(x_0) + 4h^3/3 u'''(x_0) + O(h^4) $

Where $h$ is positive and small.

We substitute the two last equations into the first one, to get:

$ u'(x_0) + R(x_0) &= (omega_0 + omega_1 + omega_2) u(x_0) + (omega_1 + 2 omega_2) h u'(x_0) \
&+ 1/2 (omega_1 + 4 omega_2) h^2 u''(x_0) + 1/6 (omega_1 + 8 omega_2) h^3 u'''(x_0) \
& + O(h^4) $

We want to minimize the residual, basicaully make $ u'(x_0) approx u'(x_0) + R(x_0)$ so $R(x_0) approx 0$.

Therefore, we can take the necessary coefficients to match them to the original function.

Namely, $ cases(
  (omega_0 + omega_1 + omega_2) = 0,
  (omega_1 + 2 omega_2) h = 1,
  1/2 (omega_1 + 4 omega_2) h^2 = 0,
) $

The solution is

$ omega_0 = -3/(2h), omega_2 = 2/h, omega_3 = -1/(2h) $

Finally, we can substitute them in


$ u'(x_0) = -3/(2h) u(x_0) + 2/h u(x_0 + h) - 1/(2h) u(x_0 + 2h) + underbrace(1/3 h^2 u''(x_0) + O(h^3), -R(x_0))$

=== Polynomial interpolation

If we want to approximate the $k^"th"$ derivative, we need to have $k + 1$ points.

$ u(x) + R(x) = P_2(x) =
  underbrace(((x-(x_0 + h)) (x - (x_0 + 2h))) / (h(2h)), l_0(x)) u(x_0) + \
  underbrace((x - x_0) (x - (x_0 + 2h)) / (h(-h)), l_1(x)) u(x_0 + h) + \
  underbrace((x - x_0) (x - (x_0 + 2h)) / (h(-h)), l_2(x)) u(x_0 + 2h) 
$

#todo[$l_2$ I think is wrong]

This is the _Lagrange_ form.

Working it out you get the same result as before.

=== Monomials

This works by enforcing the equation $x'(x_0) approx omega_0 u(x_0) + omega_1 u(x_0 + h) + omega_2 u(x_0 + 2h)$ to be exact for monomials.

Namely, those monomials are $ { 1, x, x^2, ..., x^(n-1) }$ for $n$ points. In practice, we can use ${ 1, (x - x_0), (x - x_0)^2, ...}$ to get more $0$s involved, since the spans of the two sets are the same, just with a translated basis.

Here, we replace $u(x)$ in the rhs by 
$ u(x_0) &eq.triple 1 \ =>
  omega_0 dot 1 + omega_1 dot 2 + omega_2 dot 1 &= (1)'|_(x=x_0) = 0
  \ => omega_0 + omega_1 + omega_2 = 0 $
  
$ u(x_0) eq.triple (x - x_0) => \ omega_0 (x_0 - x_0) + omega_1 (x_0 + h - x_0) + omega_2 (x_0 + 2h - x_0) = (x - x_0)'|_(x_0 = x) = 1 $

$ u(x) eq.triple (x - x_0)^2
\ => omega_0 (x_0 - x_0)^2 + omega_1 (x_0 + h - x_0)^2 + omega_2 (x_0 + 2h - x_0)^2 = ((x - x_0)^2)'|_(x=x_0)
\ => omega_1 h^2 + omega_2 4h^2 = 0
\ =>  $

== Approximations error

Let's look at it with $D_+$. The approximation error is the residual, $R(x) = D_+ u(x_0) - u'(x_0) = 1/2 h u''(x_0) + 1/6 h^2 u'''(x_0) + O(h^3)$. We can omit everything except the first term, since that overshadows the rest, so we take $1/2 h u''(x_0)$.

We define the error as $E(h) = 1/2 h u''(x_0)$, with small positive $h$ ($0 < h << 1$). Let's see how the error propagates.

#figure(table(
  columns: (auto, auto),
  $ h $, $ E(h) $,
  $ h_0 $, $ E(h_0) $,
  $ h_0/2 $, $ E(h_0 / 2) = 1/4 h_0 u''(x_0) = 1/2 E(h_0) $,
  $ h_0/4 $, $ E(h_0 / 4) = 1/8 h_0 u''(x_0) = 1/4 E(h_0) $,
  $ colon.tri.op $, $ colon.tri.op $
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
