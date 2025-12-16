#set page(numbering: "1")
#import "@preview/libra:0.1.0": balance

#align(horizon)[
  #text(size: 42pt)[Exercises Analysis] \
  Typed by Ody

  #v(1cm)

  #outline()
  #pagebreak()
]

#show heading.where(level: 1): set heading(
  supplement: "Lecture",
  numbering: "Lecture 1.",
)
#show heading.where(level: 2): set heading(supplement: "Exercise", numbering: (
  ..i,
) => [Exercise])
#show heading.where(level: 1): it => {
  set page(height: 9cm, numbering: none)
  pagebreak(weak: true)
  set text(size: 2em)
  counter(page).update(i => i - 1)
  let supple = [#it.supplement #context { counter(heading).get().at(0) }]
  let body = balance(it.body)

  align(center + horizon)[
    #grid(
      columns: (3fr, 5fr),
      align: (top + right, top + left),
      [#supple:#sym.space], body,
    )
  ]
}
#show heading.where(level: 2): it => {
  pagebreak(weak: true)
  it
}

#let exercise(it) = strong(emph(it))

= Approximation results in various of metric spaces

== 13

#exercise[
  Which complex polynomial $p(z) = p(x + i y)$ best approximates $ f(x) = cases((x y)/sqrt(x^2 + y^2) quad & (x, y) != (0, 0), 0 quad & (x, y) = (0, 0)) $
  with regard to the norm

  $
    norm(g) = abs(integral_(-oo)^oo integral_(-oo)^oo abs(g(x + i y))^2 e^(-x^2 -y^2) dif x dif y)^(1/2)
  $
]

We take space

$
  X = { g in C(RR) : norm(g) < oo }
$

and innerproduct

#let innerproduct(f, g) = math.lr($angle.l #f, #g angle.r$)

$
  innerproduct(f, g) = integral_(-oo)^oo integral_(-oo)^oo f(x + i y) overline(g(x + i y)) e^(-x^2 - y^2) dif x dif y
$

We want to find an orthonormal basis $g_(j, j') (x, y) = T_j (theta) R_j' (r)$, such that

#let intoo = $integral_(-oo)^oo$
$
  delta_(j, k) & = intoo intoo T_j (theta) R_j (r) overline(R_k (theta) R_k (r)) e^(-r^2) dif x dif y \
  &= integral_0^(2pi) intoo T_j (theta) overline(T_k (theta)) R_j (r) R_k (r) r e^(-r^2) dif r dif theta
$

We need to choose something for $T_j$. Something reasonable seems like $T_j (theta) = e^(i j theta)$.

$
  & = integral_0^(2pi) e^(i (j - k) theta) dif theta intoo R_(j') (r) overline(R_(k')(r)) r dif r
$

for $j != k$ this is $0$. For $j = k$:

$
  & = 2pi intoo abs(R_j' (r))^2 dif r
$

So, how do we make $R_j' (r) e^(i j theta)$ a complex polynomial? Well, we need to have $R_j (r) = r^j$ then we get that
$
  g_(j, j) (x, y) = r^j e^(i j theta) = (r e^(i theta))^j = (x + i y)^j
$

that is, $g_j (x, y) = (x + i y)^j$.

Well, technically we need to take $R_j (r) = r^j/sqrt(pi j!)$ because

$
  2pi intoo abs(R_j' (r))^2 e^(-r^2) dif r & = 2pi intoo r^(2j + 1) e^(-r^2) dif r \
  & = 2pi intoo r^(j + 1/2) e^(-r) (dif r)/(2 sqrt(r)) = (2pi)/2 Gamma(j + 1) \
  & = pi j!
$

and we want to make it orthonormal.

The basis we have is ${ g_0, g_1, g_2, ... }$ which forms an orthonormal system.

Now we can use theorem 1.9 to find the best approximation using projection. Note that $f$ is nice in polar coordinates:

$
  f(x, y) = (x y)/sqrt(x^2 + y^2) = (r^2 cos theta sin theta)/r
$

So we can calculate the innerproducts in polar coordinates

$
  innerproduct(f, g_j) = intoo intoo f(x + i y) overline(g_j (x + i y)) e^(-x^2 - y^2) dif x dif y
  & = integral_0^(2pi) intoo (r^2 cos theta sin theta)/r r^j e^(--i j theta) e^(-r^2) r dif r dif theta \
  & = 1/(4i) integral_0^(2pi) (e^(2 i theta) - e^(- 2 i theta)) e^(-i j theta) dif theta intoo r^(j + 2) e^(-r^2) dif r
$

$e^(-2i theta)$ always contributes $0$, and $e^(2 i theta)$ is nonzero only if $j = 2$. So, the only nonzero inner product will be

$
  innerproduct(f, g_2) = (2pi)/(2i) intoo r^(j/2 + 1) e^(-r) (dif r)/(2 sqrt(r))
  & = pi/(4i) Gamma(5/2) \
  & = pi/(4i) 3/2 1/2 sqrt(pi) \
  & = (2pi)/(16 sqrt(2)) (x + i y)^2
$

Since all the other inner products are $0$, this is the best approximation to $f$ for any $n >= 2$ (and for $n < 2$, the best approximation is $0$).

= Best approximations and convexity

== 22

#exercise[
  Prove that there exist coefficients $a_k$ such that $T_n(x) = sum_(k=0)^n a_k x^k a_k (cos theta)^k$ for $k=0, ..., n$.
]

We are going to use Euler's identity and binomial formula:

$
  cos^k theta & = (cos theta)^k \
  & = ((e^(i theta) + e^(-i theta))/2)^k \
  & = 1/(2^k) sum_(j=0)^k (e^(i theta))^j (e^(-i theta))^(k - j) mat(k; j) \
  & = 1/(2^k) sum_(j=0)^k mat(k; j) cos((2j - k) theta)
$

On the last step we used the fact that $cos^k theta$ is real, so we can just take the real part of the complex sum (yes, it does work out that the real part of a sum is the sum of the real parts).

Now we can show the original statement by induction. For $n = 0$ it's kind of obvious, since just $a_(0, 0) = 1$.

Now assume the statement is true for up to $n - 1$. Then if we look at $2 cos(n theta)$we see that it is last and the first term of the sum above (for $k=n$). So,

$
  2/2^n cos(n theta) & = cos^n(theta) - 1/(2^k) sum_(j=1)^(n-1) mat(n; j) cos(abs(2j - n) theta) \
  & = cos^n(theta) - 1/(2^k) sum_(j=1)^(n-1) mat(n; j) sum_(ell=0)^abs(2j - n) a_(abs(2j - n), ell) cos^ell (theta)
$

Using the fact that $abs(2j - n) <= n - 2 < n$ we can write that

$
  2/2^n cos(n theta) = sum_(k=0)^n a_(n, k) cos^k (theta)
$

since the coefficients will not exceede $n$, and so we've shown that there exists this sum.

And, as a bonus, we see that the leading coefficient of the Chebyshev polynomial is $2^(n - 1)$

== 26

#exercise[
  Let $P$ be a polynomial satisfying
  $ P(x_i) = f(x_i), P'(x_i) = f'(x_i) $

  for some function $f$ and points $x_1 < x_2 < ... < x_n$.

  + Show that $P$ is unique (if exists)
  + Prove that $P$ is given by $ P_n (x) = sum_(j=1)^n (f(x_j) ( 1 - alpha_j (x - x_j)) + f'(x_j) (x - x_j)) (W(x)^2)/((x-x_j)^2 W'(x_j)^2) $

  where $W$ is the wronskian $W(x) = product_(k=1)^n (x - x_k)$.
]

+ Suppose that $tilde(P)$ is also a polynomial satisfying these conditions. We will show that $P = tilde(P)$. If we take the difference $Q(x) = P(x) - tilde(P) (x)$ we see that

  $
     Q(x_i) & = f(x_i) - f(x_i) = 0 \
    Q'(x_i) & = f'(x_i) - f'(x_i) = 0
  $

  Notice that this implies that at the interpolation points we have two zeros, since the value and the derivative is $0$ so $Q(x)$ around $x_i$ is $"constant" (x - a)^2$.

  Then, we can write

  $
    Q(x) = q(x) product_(i=1)^n (x - x_i)^2
  $

  But these are $2n$ zeros while the degree of $Q$ is at most $2n - 1$, so this can only happen is $Q$ is identically $0$.

+ Let's inspect $ lim_(x -> x_j) W(x) / ((x-x_j) W'(x_j)) $
  This is well defined because we have a simple $0$ at $x_j$:
  $
    lim_(x -> x_j) (W(x) - W(x_j))/(x - x_j) 1/(W'(x_j)) = (W'(x_j)) / (W'(x_j)) = 1
  $

  and for any other point
  $
    lim_(x -> x_i \ i != j) (W(x))/((x-x_j) W'(x_j)) = lim_(x -> x_i) product_(k=1 \ k != j)^n (x - x_k) 1/ (W'(x_j)) = 0
  $

  We are going to use these properties. But let's start with developing $P_n(x_i)$:

  $
    P_n (x_i) = f(x_i) ( 1 - cancel(alpha_i (x_i - x_i))) + cancel(f'(x_i) (x_i - x_i)) dot 1 = f(x_i)
  $

  And the derivative $P'_n(x_i)$ (this is a bit more involved):

  $
    P'_n (x) & = sum_(j=1)^n (0 - alpha_j f(x_j) + f'(x_j)) W(x)^2/((x-x_j)^2 W'(x_j)) \
    & + sum_(j=1)^n (f(x_j) (1 - a_j (x - x_j)) \
      & + f'(x_j) (x - x_j)) sum_(k=1 \ k!=j) 2(x - x_k) product_(ell=1 \ ell!=j,k)^n (x - x_ell)^2 \
    P'_n(x_i) & = -alpha f(x_i) + f'(x_i) + (f(x_i) dot 1 + 0) lim_(x->x_i) sum_(k=1 \ k != i)^n 2(x - x_k) product_(ell = 1 \ ell != i, k) (x - x_ell)^2 1/(W' (x_ell)^2) \
    & = -alpha f(x_i) + f'(x_i) + (f(x_i) dot 1 + 0) (2(n - 1))/(W'(x_i)) \
    & = f'(x_i) \
  $

  And this holds if $alpha_i = (2(n-1))/(W'(x_i))$.

= Topology and $RR$

== 34

#exercise[
  Show that $integral_[0,1] f dif m = 1/2$ for $f(x) = x$, using simple functions.
]

We are going to take

$
  S_n (x) & = cases(k/n) quad "for" quad x in [k/n, (k+1)/n) \
          & = sum_(k=0)^(n-1) k/n chi_[k/n, (k+1)/n)
$

So then,

$
  integral_[0, 1] S_n dif m & = sum_(k=0)^(n-1) k/n integral_[0, 1] chi_[k/n, (k+1)/n) (x) dif m (x) \
  & = sum_(k=0)^(n-1) k/n 1/ n \
  & = 1/n^2 sum_(k=0)^(n-1) k \
  & = 1/n^2 ((n-1) n)/2 \
  & = 1/2 - 1/(2n) \
$

So then $integral_[0, 1] f dif m >= 1/2 - 1/(2n)$. To find an upper bound we can consider $tilde(f) = 1 - f$. We can do the same reasoning with $tilde(S)_n (x) = 1 - (k+1)/n quad x in [k/n, (k+1)/n)$, where we deduce that $integral_[0, 1] tilde(f) dif m >= 1/2 - 1/(2n)$, which implies that

$
  integral_[0, 1] f dif m = 1 - (1/2 - 1/2m) = 1/2 + 1/(2m)
$

== 36

#exercise[
  Let $B$ be the subset of $(0, 1)$ of elements that have no $9$ in their decimal expansion. Is $B$ a Borel set?
]

Turns out it is. We will show this.

Let $ B = inter_(n=1)^oo B_n quad "where" B_n = { x in (0, 1): "the" n^"th" "decimal is not a" 9 } $

This is a countable intersection, so it suffices to show that all $B_n$ are Borel sets which in turn can be shown if the complements $(B_n)^c$ are Borel. This is easier to show because $(B_n)^c$ are the numbers where the $n^"th"$ decimal digit _is_ a $9$.
This can be written as:

$
  (B_n)^c & = { x in (0, 1): n^"th" "decimal is" 9 } \
  & = { sum_(j=1)^(n-1) a_j 10^(-j) + 9 dot 10^(-n) + sum_(j=n+1)^oo a_j 10^(-j) : a_j in {0, 1, ..., 9} quad ("but not all are" 9)}
$

We want to rewrite this as a union.

$
  10^(-n) sum_(j=1)^(n-1) (a_j 10^(n - j)) + 10^(-n) dot 9 + 10^(-n) underbrace(sum_(j=1)^oo a_(j + n) 10^(-j), <= 1)
$

We define $k$ as

$
  k = sum_(j=1)^(n-1) (a_j 10^(n-j)) = sum_(j=1)^(n-1) a_(n - j) 10^j
$

The biggest value of $k$ we can reach is:

$
  =9 (10^n - 10)/(10 - 1) = 10^n - 10
$

So then we can write $B_n$ as
$
  (B_n)^c = union_(k=0)^(10^n - 10) [k/10^n + 9/10^n, (k+1)/n)
$

= Best approximations in $norm(dot)_oo$

== 46

#exercise[
  Remember the Hermite polynomials
  $
    H_j (x) = (-1)^j e^x^2 dif^j /(dif x^j) (e^(-x^2))
  $

  + Write $f(x) = sin x$ as a series in $H_j$.
  + What is the error of the $n^"th"$ order partial sum?
  #v(0cm)
]

+ From exercise $8$ we know that $H_j$ is orthogonal with inner product:

  $
    innerproduct(f, g) & = intoo f(x) g(x) e^(-x^2) dif x
  $

  We know that $g_j c_j H_j$ forms an ortho#emph[normal] basis with $c_j = 1/(2^(j/2) sqrt(j!) pi^(1/4))$. Then, using $sin x = sum_(j=0)^oo innerproduct(f, g_j) g_j (x)$.

  $
    innerproduct(f, g_j) & = intoo sin(x) c_j H_j (x) e^(-x^2) dif x (-1)^j/(-c_j) intoo sin(x) e^(x^2) dif^j/(dif x^j) (e^(-x^2)) e^-x^2 dif x \
    & = (-1)^j c_j (-1)^j intoo sin^((j)) (x) e^(-x^2) dif x
  $

  If $j$ is even, $innerproduct(f, g_j) = 0$ by symmetry. If $j$ is odd,

  $
    innerproduct(f, g_j) & = (-1)^((j-1)/2) c_j intoo cos(x) e^(-x^2) dif x = (-1)^((j-1)/2) c_j sqrt(pi) e^(1/4) \
    & = cos(x) e^(-x^2) dif x \
    & = Re intoo e^(-x^2 + i x) dif x \
    & = Re intoo e^((-x -i/2)^2 - 1/4) dif x
  $

  Define

  $
    I(t) & = intoo e^(-(x - t)^2A dif t) \
         & = intoo e^(-x^2) dif x = sqrt(pi)
  $

  So $innerproduct(f, g_j) = Re sqrt(pi) e^(-1/4) = sqrt(pi) e^(-1/4)$.

+ The error is given by

  $
    sqrt(sum_(k=floor(n/2)) abs((-1)^k c_(2k + 1) sqrt(pi) e^(-1/4))^2)
  $

  And it's a bit of a pain in the ass to show but apparently it is exponentially accurate.

== 47

#exercise[
  Supopose ${q_j }$ is a basis of monic polynomials of degree $j$ which are orthogonal with respect to $w$:

  $
    integral_[a, b] q_i (x) q_j (x) w(x) dif m (x) = delta_(i, j)
  $

  Prove that $ q_j = 1/D_j abs(
    mat(
      m_0, m_1, ..., m_j;
      m_1, m_2, ..., m_(j+1);
      dots.v;
      m_(j-1), m_j, ..., m_(2j - 1);
      1, x, ..., x^j;
    )
  ) $

  where $ D_j = abs(
    mat(
      m_0, m_1, ..., m_(j-1);
      m_1, m_2, ..., m_j;
      dots.v;
      m_(j-1), m_j, ..., m_(2j - 2)
    )
  ) $

  and where the $m_k$s are the "moments", defined as:

  $
    m_k = integral_[a, b] x^k underbrace(w(x) dif m(x), dif mu)
  $
]

First, let's show it is a monic polynomial. If we develop the determinant

$
  q_j (x) = 1/D_j (x^j D_j + ...) = x^j + ...
$

we see that it is indeed monic.

Now let's try to prove the main statement. Suppose without loss of generality that $i < j$. Then, it is enough to show the orthogonality condition with $q_i (x)$ replaced with $x^i$, since $q_i$ is a polynomial. That is,

$
  & D_j integral x^i q_j (x) w(x) dif m(x) \
  & = integral x^i abs(
      mat(
        m_0, m_1, ..., m_j;
        m_1, m_2, ..., m_(j+1);
        dots.v;
        m_(j-1), m_j, ..., m_(2j - 1);
        1, x, ..., x^j;
      )
    ) \
  & = abs(
      mat(
        m_0, m_1, ..., m_j;
        m_1, m_2, ..., m_(j+1);
        dots.v;
        m_(j-1), m_j, ..., m_(2j - 1);
        integral x^i dif mu, integral x^(i+1) dif mu, ..., integral x^(j+i) dif mu;
      )
    ) \
  & = abs(
      mat(
        m_0, m_1, ..., m_j;
        m_1, m_2, ..., m_(j+1);
        dots.v;
        m_(j-1), m_j, ..., m_(2j - 1);
        m_i, m_(i+1), ..., m_(i + j)
      )
    ) \
  & = 0.
$

In the last step we used the fact that the last row is the same as the $i^"th"$ row, which means that the matrix is linearly dependent and thus its determinant has to be $0$.

= Best approximations in $L^1$

== 54

#exercise[
  Let $h_n$ be the best monic polynomial of degree $n$ that approximates to the $0$ function in $L^1 [-1, 1]$ sense. Show that $h_n = 2^(-n) U_n$, where
  $
    U_n (cos theta) = sin((n+1) theta)/sin(theta) quad "for" med theta in (0, pi)
  $

  ($U_n$ is the Chebychev polynomial of the $2^"nd"$ kind).
]

We are going to use characterization theorem (theorem 54). Therefore, we need to show that:

#let sgn = $"sgn"$

$
  integral_(-1)^1 sgn(0 - U_n (x)) x^j dif x = 0 quad quad forall j = 0, ..., n-1
$

First, let's show that $U_n$ is a monic polynomial. Given the definition,

$
         U_n (cos theta) & = sin((n+1)theta)/sin(theta) \
  2^(-n) U_n (cos theta) & = 2^(-n) sin((n+1)theta)/sin(theta) \
$

we want to show that the leading term in a polynomial expansions of powers of cosines has degree $1$. We can do this by rewriting as complex exponentials:

$
  2^(-n) U_n (cos theta) & = 2^(-n) sin((n+1)theta)/sin(theta) \
  & = 2^(-n) (cancel(1/(2i)) (e^(i(n+1) theta) - e^(-i(n+1)theta)))/(cancel(1/(2i)) (e^(i theta) - e^(-i theta))) \
  & = 2^(-n) (e^(2i (n+1) theta) - 1)/(e^(2i theta) - 1) (e^(-i (n+1)theta))/(e^(-i theta)) \
  ("geometric series formula") quad & = 2^(-n) sum_(k=0)^n (e^(2i theta))^k e^(-i n theta) \
  & = 2^(-n) sum_(k=0)^n e^(i (2k - n) theta) \
  ("taking the real part") quad & = 2^(-n) sum_(k=0)^n cos((2k - n) theta) \
  & = 2^(-n) sum_(k=0)^n T_abs(2k - n) (cos theta) \
  & = 2^(-n) (T_n (cos theta) + T_n (cos theta) + ...) \
  & = 2^(-n) (2^(n-1) + 2^(n-1)) cos^n theta + ... \
  & = cos^n theta + ...
$

so it is indeed monic!

Now we want to see whether it is the best approximation. We know that $sin((n+1) theta)$ changes sign at exactly $theta = pi/(n+1) k$ for $k=1, ..., n$. Using the Chebyschev polynomials of the first kind we know we can write $(cos theta)^j$ as linear combinations of $cos(j theta)$ so we only have to show that those are orthogonal. Then, we can also use the identity $cos(j theta) sin(theta) = 1/2 sin((j+1) theta) - 1/2 sin((j-1) theta)$ to get another equivalent expression. That is, it is sufficient to show:

$
  & integral_0^(2pi) sgn((sin((n+1)theta))/sin(theta)) (cos(theta))^j sin(theta) dif theta = 0 \
  <== & integral_0^(2pi) sgn((sin((n+1)theta))/sin(theta)) cos(j theta) sin(theta) dif theta = 0 \
  <== & integral_0^(2pi) sgn((sin((n+1)theta))/sin(theta)) sin(j theta) dif theta = 0
$

And this last one we can solve, using again complex exponentials and taking the real part:

$
  & integral_0^(2pi) sgn((sin((n+1)theta))/sin(theta)) sin(k theta) dif theta \
  = & sum_(j=0)^n integral_((pi j)/(n+1))^((pi(j+1))/(n+1)) (-1)^j sin(k theta) dif theta \
  = & -1/k sum_(j=0)^n (cos((pi k (j + 1))/(n+1)) - cos((pi k j)/(n+1))) (-1)^j \
  = & -1/k Re(sum_(j=0)^n (e^((pi k (j + 1))/(n+1)) - e^((pi k j)/(n+1))) (-1)^j) \
  = & -1/k Re((e^((i pi k)/(n+1)) - 1) sum_(j=0)^n (e^((i pi k j)/(n+1)) ) (-1)^j) \
  = & -1/k Re((e^((i pi k)/(n+1)) - 1) sum_(j=0)^n (-e^((i pi k)/(n+1)))^j) \
  = & -1/k Re((e^((i pi k)/(n+1)) - 1) (e^((i pi k j))^(n+1) - 1)/(-e^((i pi k)/(n+1)) - 1)) \
  = & -1/k ((-1)^((k+1)(n+1)) - 1) Re(i tan((pi k)/(2(n+1)))) \
  ("since" tan("real") = "real") quad = & -1/k ((-1)^((k+1)(n+1)) - 1) dot 0 \
  = & 0
$

which, despite being very ugly, implies orthogonality as we wanted to show.

== 63

#exercise[
  Let $m$ be Lebesgue measure on $[0, 1]$. Let weight $w(x) = (x+1)/2$.  We define $A$ as

  $
    A = { g in L^1([0, 1], m) : integral_0^1 g w dif m = 0 }.
  $

  Let $f in L^1([0, 1], m) \\ overline(A)$. Prove that $inf norm(f - g)_1 = abs(integral_0^1 f w dif m)$ and that $f$ has no best $L^1$ approximant in $A$.
]

Let's write out $norm(f - g)$, for any $g in A$:

$
  norm(f - g) = integral_0^1 abs(f(x) - g(x)) dif m(x)
$

If we didn't have the absolute value we could do something with this integral, which hints we should use the inverse triangle inequality. But, before that, we are going to insert the weight (because the weight is $< 1$).

$
  norm(f - g) & = integral_0^1 abs(f(x) - g(x)) dif m(x) \
  & >= integral_0^1 abs(f(x) - g(x)) \
  & >= abs(integral_0^1 (f(x) - g(x)) w(x) dif m (x)) \
  & = abs(integral_0^1 f(x) w(x) dif m(x) - integral_0^1 g(x) w(x) dif m (x))
$

but, since we know that $g in A$, we have that $integral_0^1 g(x) w(x) dif m (x)$, so

$
  norm(f - g) & >= abs(integral_0^1 f(x) w(x) dif m(x) - 0)
$

so indeed we see that $abs(integral_0^1 f(x) w(x) dif m(x))$ is a valid lower bound, which for brevity we are going to call $I$.

Now we want to show now that $abs(I)$ is the _maximum_ lower bound, meaning that some $g in A$ achieves it. But actually no single $g in A$ achieves this (proving this is part 2 of the question), so we have to show that some sequence $g_n$ has a limit in $A$ that achieves this value. To do this we are going to take a sequence $h_n$:

$
  h_n (x) = n chi_(1 - 1/n, 1) (x)
$

where the idea is that we are taking a region around where the weight is $1$, because if the weight was $1$ it would be easy to show that it holds. If we calculate the condition of being in $A$:

$
  integral_0^1 h_n dif m = 1
$

we see that $h_n in A$.

Now, we take sequence

$
                   g_n & = f - c_n h_n \
  integral_0^1 w dif m & = integral_0^1 (f - c_n h_n) w dif m \
                       & = I - c_n integral_0^1 h_n w dif m \
                       & = I - c_n integral_(1-1/n)^1 n (x + 1)/2 dif x \
                       & = I - c_n n [(x+1)^2/4]_(1-1/n)^1 \
                       & = I - c_n (1 - 1/(4n)) \
$

we need this last expression to be $0$ for $g_n$ to belong in $A$, so we set $c_n = (1 - 1/(4n))^(-1) I$. Now, if we calculate

$
  norm(f - g_n)_1 & = integral_0^1 abs(f(x) - g_n (x)) dif m(x) \
                  & = integral_0^1 abs(c_n) abs(h_n (x)) dif m(x) \
                  & = abs(c_n) dot 1 \
                  & = abs((1 - 1/(4n))^(-1)) I \
$

since $g_n in A$ for any $n$, we can take the limit as $n -> oo$ so we find that

$
  inf norm(f - g) <= norm(f - g_n) <= abs((1 - 1/(4n))^(-1)) abs(I) \
  => inf norm(f - g) <= abs(I)
$

and thus we have successfully sandwiched the infimum, so we can conclude that

$
  inf norm(f - g) = abs(I)
$

*Showing no best approximation exists*

To show this, we can use contradiction: assume there is some $g$ such that

$
  norm(f - g) & = abs(I) \
              & = abs(integral_0^1 f w dif m) \
              & = abs(integral_0^1 (f - g) w dif m) \
              & <= integral_0^1 abs(f(x) - g(x)) w(x) dif m(x) \
              & <= integral_0^1 abs(f(x) - g(x)) dot 1 dot dif m(x) \
$

This implies that $abs(f - g) w = abs(f - g) dot 1$ almost everywhere. However, we know that $w < 1$ almost everywhere, so we have to have that $f = g$ almost everywhere. But if this is the case then $f$ would need to be in $A$, which we deny. Therefore we have a contradiction and we conclude that there is no best approximation.

= Splines

== 73

#exercise[
  Let $P$ be a linear spline interpolating to $f$ at $x_0 < x_1 < ... < x_n$. Prove that

  $
    norm(f - p)_oo <= 1/8 norm(f'')_oo (max Delta i)^2
  $

  where $Delta i = Delta x_i = x_(i+1) - x_i$.
]

We have an explicit equation for linear splines. Namely, on $[x_i, x_(i+1)]$ we have that
$
  p(x) = f(x_i) (x_(i+1) - x)/Delta_i + f(x_(i+1)) (x - x_i) / Delta_i
$

The idea is that $norm(f'')_oo$ gives us a bound on how much $f$ can vary in an interval. We can also think of the mean value theorem, but we don't have two values of the derivatives being $0$, only one.

Here's the trick. We start with function $phi_0 (y) = f(y) - p(y)$. We want this function to have one more zero. Notice that we can add $(y - x_i)(x_(i+1) - y)$ to $phi_0$ and still have zeros at $x_i$ and $x_(i+1)$. So,

$
  phi(y) = f(y) - p(y) - lambda (y - x_i) (x_(i+1) - y)
$

Now we want to generate a new $0$. We already have that $phi(x_i) = phi(x_(i+1)) = 0$. Now, if we choose $lambda = (f(x) - p(x))/((x-x_i)(x_(i+1) - x))$ we can find two values $alpha$ and $beta$ such that $phi'(alpha) = phi'(beta) = 0$. This is now enough to show that there exists a value $gamma in (alpha, beta)$ where $phi''(gamma) = 0$.

If we now try to evaluate $phi''(gamma)$ we get

$
  phi''(gamma) = f''(gamma) + underbrace(cancel(p''(gamma)), "since" p "is linear") - 2 (f(x) - p(x))/((x-x_i)(x_(i+1) - x))
$

If we now try to write a bound for $abs(f(x) - p(x))$ we can find that

$
  abs(f(x) - p(x)) <= 1/2 norm(f'')_oo (x - x_i) (x_(i+1) - x)
$

we can rewrite $(x - x_i) (x_(i+1) - x)$ as $-(x - (x_i + x_(i+1))/2) + 1/4 (Delta x_i)^2$ which is of course bounded by $1/4 (max Delta_j)^2$, and thus:

$
  abs(f(x) - p(x)) <= 1/2 norm(f'')_oo (x - x_i) (x_(i+1) - x) <= 1/8 norm(f'')_oo (Delta x_i)^2
$

== 77

#exercise[
  Given equispaced points $Omega_n$ where $Delta x_i = (b - a)/(n+1)$. Determine what is the inverse of

  $
    A =
  $
]

The exercise has a hint of considering a more general matrix:

$
  A = mat(
    z, 1, 0, ..., 0;
    1, z, 1, dots.down, dots.v;
    0, 1, z, dots.down, 0;
    dots.v, dots.down, dots.down, dots.down, 1;
    0, ..., 0, 1, z;
  )
$

The idea is that we can solve it recursively. Specifically we want to find solutions of the type $A x = e_j$ where $e_j$ is the $j^"th"$ unit vector.

Let's first try an example of a $5 times 5$ matrix:

$
  mat(
    z, 1, , , , 0;
    1, z, 1;
    , 1, z, 1;
    , , 1, z, 1;
    0, , , 1, z, 1;
  ) mat(x_1; x_2; x_3; x_4; x_5) = mat(
    x_0 + 2 x_1 + x_2;
    x_1 + 2x_2 + x_3;
    x_1 + 2x_2 + x_3;
    x_2 + 3x_3 + x_4;
    x_3 + 4x_4 + x_5;
    x_4 + 5x_5 + x_6;
  ) = mat(0; dots.v; 1; dots.v; 0)
  quad ("this is the" j^"th" " element")
$

$x_0$ and $x_(n+1)$ are not actually variables we have, so we can define them to what's convenient. Namely, $x_0 = x_(n+1) = 0$.

So, for all $i != j$ we have that

$
  x_(i-1) + z x_i + x_(i+1) & = 0 \
$

and for $j$ we have

$
  x_(j-1) + z x_j + x_(j+1) & = 1 \
$

These are linear recurrances. The first one has characteristic equation (???)

$
  lambda^2 + z lambda + 1 = 0 => lambda_plus.minus = 1/2 (-z plus.minus sqrt(z^2 - 4))
$

so then (???) we get $ cases(
  x_i = A lambda_+^i + B lambda_-^i quad & "for" i = 0\, 1\, ...\, j,
  x_i = C lambda_+^(n+1-i) + D lambda_-^(n+1-i) quad & "for" i = j+1\, ...\, n+1
) $

Given $x_0 = 0 = A + B$ we deduce $B = -A$ and with $x_(n+1)$ we get $C+ D = 0 => D = -C$. Using relations between the equations we can find that

$
  A(lambda_+^j - lambda_-^j) = C(lambda_+^(n+1-j) - lambda_-^(n+1-j)) \
  => C = A(lambda_+^j - lambda_-^j)/(lambda_+^(n+1-j) - lambda_-^(n+1-j)) \
$

And finally, we can use some other equation (idk which) to get

$
  1 = A(lambda_+^(j-1) - lambda_-^(j+1)) ...
$

(look, imma be honest, I just stopped copying here)

As a fun fact, if you take $z = 2 cos theta$ you get complex exponentials and you can rewrite the equations using Chebyshev polynomials of the second kind.

= Padé approximants

== 83

#exercise[
  In the case of generalized Padé approximants of order $(n, m)$ the monomials are replaced by Chebyshev polynomials.

  Prove that the equations of Theorem 7.1 and 7.2 are replaced by

  $
    1/2 sum_(i=0)^oo (a_(k+i) q_i + a_i q_(k+i) + a_i q_(k+i)) - p_k = c_k quad "for" k=1,2,... \
    1/2 (a_0 q_0 + sum_(i=0)^m a_i q_i) - p_0 = c_0
  $

  where
  $
    f = sum_(i=0)^oo a_i T_i = P/Q + R/Q \
    P = sum_(i=0)^oo p_i T_i \
    Q = sum_(i=0)^oo q_i T_i \
    R = sum_(i=0)^oo c_i T_i \
    c_(n+m+1 - i) = p_(n+i) = q_(m+i) = q_(-i) = p_(-i) = a_(-i) = c_(-i) = 0 quad "for" i = 1, 2, ...
  $
]

We are going to use that in exercise $82$ we show that $ T_i T_j = 1/2 T_(i+j) + 1/2 T_abs(i - j) $


Theorem 7.3 tells us that
$
  p_k = sum_(i=0)^oo sum_(j=0)^oo A_(i j k) a_i q_j
$

and we also have that $T_i T_j = sum_(k=1)^(i+j) A_(i j k) T_k$ and respectively for $P$.

From the result of exercise $82$ we can deduce when $A_(i j k)$ is nonzero. Namely:

$
  A_(i j k) = cases(
    1/2 quad & "if" k = i + j,
    1/2 quad & "if" k = plus.minus abs(i + j),
    0 quad & "otherwise",
  )
$

If we replace this expression in theorem 7.3 we get

$
  p_k = 1/2 sum_(i+j=k) a_i q_j + 1/2 sum_(abs(i-j) = k) a_i q_j
$

and if we define $i - j = plus.minus k$ so that $j = i minus.plus k$ to get

$
  p_k = 1/2 sum_(i=0)^oo a_i q_(k-i) + sum_(i=0)^oo (a_i q_(i+k) + underbrace(a_i q_(i-k), a_(i+k) q_i))
$

and from here it's not too hard to show that you can shift around the indices to get the expression we're looking for.
