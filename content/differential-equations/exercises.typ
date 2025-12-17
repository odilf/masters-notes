#import "@preview/lilaq:0.5.0" as lq

#show heading.where(level: 1): set heading(numbering: "1.")
#let faint = it => {
  set text(fill: white.darken(60%))
  strong(it)
}

#set math.mat(delim: "[")

#show heading.where(level: 2): it => {
  v(2cm)
  it
}

#align(horizon)[

  #text(size: 42pt)[Techniques for Differential Equations]
  #v(7mm, weak: true)
  #text(size: 28pt)[Summary and exercises]
  #v(1cm)

  #outline()

  #v(1cm)
  #faint[In the final he's going to asks us problems like this. One of the problems could be "consider this problem: solve it" or "write a numerical scheme to solve it" and we might have to write some pseudocode on how to solve this.]
]

#pagebreak(weak: true)

= Finite differences

#faint[== Exercise 3]
How to interpolate?
$
  u''(x_0) approx a u(x_0 - h) + b u(x_0) + c u(x_0 + h)
$

We can do it three ways:
+ Taylor expansions
+ Lagrange interpolation
+ Enforce exact for ${ x, x, x^2 }$

What if the points are not equispaced? We can still apply it the same way, just changing the evaluation points.

What is the local truncation error? The difference between the real value and the formula you use. Generally we find it using Taylor:

$
  tau(x_0) = underbrace(L u, u'') (x_0) - sum_i omega_i u(x_i)
$

which we expect to be of order $k$, $O(h^k)$, where $k$ depends on the approximation formula.

If we plot the error, we expect it to decrease as a straight line in a log-log plot with slope $k$.

#{
  let x = lq.linspace(1, 6, num: 10).map(i => 100 + calc.pow(10, i))

  lq.diagram(
    lq.plot(x, x.map(x => calc.pow(x, 2))),
    xscale: lq.scale.log(),
    yscale: lq.scale.log(),
    xlabel: $h$,
    ylabel: [Error],
  )
}

= Boundary value problems

Problems of type

$
  u'' = f, x in I \
  u(a) = u(b) = 0
$

We can solve using finite differences. Specifically, FD2:

$
  (dif^2 u)/(dif x^2) = 1/h^2 ( u_(i-1) - 2u_i + u_(i+1)) = f_i \
  u_0 = u_(m+1) = 0
$

This gives matrix

$
  A arrow(u) = arrow(f)
$

which we need to solve, which gives $U_i approx u(x_i)$.

We want our methods to be _convergent_ which means they are _consistent_ + _stable_:

- Consistent means that $tau -> 0$ as $h -> 0$
- Stable means the error $norm(E) -> 0$ as $h -> 0$.
- Convergent is consistent + stable

Given $A arrow(E) = arrow(tau)$ where $E_i = U_i - u(x_i)$ and $tau(x_i) = L u_i - "FD"(x_i)$ we have a result that

$
  norm(A_h^(-1))_2 <= C, forall h
$

where $C$ is some constant.

#faint[== Exercise 2]

We have that $u_p = sin(p pi x)$ are eigenfunction of $diff^2/(diff x^2)$ with eigenvalues $lambda_p = -p^2 pi^2$. That is,

$
  (diff^2 u_p)/(diff x^2) = -p^2 pi^2 u_p = lambda_p u_p
$

The analogous for the discrete case is that

$
  [A arrow(u)]_j & = (u_(j+1) - 2u_j + u_(j+1))/h^2 \
                 & = 1/h^2 (
                     sin((j+1) p pi h) - w sin(j p pi h) + sin((j-1) p pi h)
                   ) \
                 & = 1/h^2 (2 sin(j p pi h) cos(p pi h) - 2 sin(j p pi h)) \
                 & = sin (j p pi h) (2/h^2 (cos(p pi h) - 1)
$

So here the eigenfunction is $sin(j p pi h)$ and eigenvalue is $lambda_p = 2/h^2 (cos(p pi h) - 1)$

What does this have to do with the error? Well, the error is related to the norm of the inverse which is given by the spectral radius:

$
  norm(A_h^(-1))_2 <= C, forall h \
                 norm(A_h^(-1))_2 & = rho(A^(-1)) \
$

The smallest eigenvalue from before is
$
  lambda_1 = 2/h^2 (cos(pi h) - 1) ->_(h -> 0) 2/h^2(1 + (pi h^2) - 1) = 2pi^2
$

So then

$
  norm(A_h^(-1))_2 = rho(A^(-1)) = 1/(2pi^2) <= C, forall h
$

Therefore the method is convergent.

= BVD in 2D

Problems of the type

$
  laplace u = f, overline(x) in Omega \
  u|_(diff Omega) = g
$

We can solve this using the 5-point Laplacian:

$
  laplace_5 u_(i j) = 1/h^2 (
    4 u_(i, j) + u_(i+1, j) + u_(i-1,j) + u_(i, j+1) + u_(i, j-1)
  )
$

We can also use the 9-point Laplacian. This doesn't give more accuracy directly, but it gives an error term of $h^2/12 laplace^2 f$ which can be calculated from $laplace u$, which we know. Therefore, we can correct for that error and get accuracy $O(h^4)$.

#counter(heading).step()
#counter(heading).step()

= Parabolic PDEs

$
  U_i^(n+1) = alpha u_i^n + (1 - alpha)/2 (U_(i+1)^n + U_(i-1)^n) \
  alpha = 1 - 2 beta mu \
  mu = k / h^2
$

This is this consistent (I think?)

We want to find the order.

$
  U_i^(n+1) & = (1 - 2 beta mu) u_i^n + (1 - alpha)/2 (U_(i+1)^n + U_(i-1)^n) \
  & = u_i^n - 2 beta mu u_i^n + beta mu U_(i+1)^n + beta mu U_(i-1)^n \
  => (U_i^(n+1) - U_i^n)/k & = 1/h^2 beta (u_(i-1) - 2 u_i^n + u_(i+1)^n) \
$

This is the equation $u_t = beta u_(x x)$ using FE and FD2 so we expect that $tau = O(k + h^2)$. And if we write it out:

$
           u(x, t + k) & = u + k u_t + k^2/2 u_(t t) + O(k^3) \
  => (u^(n+1) - u^n)/2 & = u_t + k/2 u_(t t) + O(k^2)
$

and

$
  u(x + h) + u(x - h) = 2u + h^2 u_(x x) + h^4/12 u_(x x x x) + ... \
  => (u_(i+1) - 2u_i + u_(i-1))/h^2 & = (u_(x x))_i + h^2/12 (u^(("iv")))_i + ...
$

And if we substitute those Taylor expansions in:

$
  tau & = (u(x_i, t_(n+1)) - u(x_i, t_n))/k - beta/h^2 ( u(x_i, t_n) - 2 u(x_i, t_n) + u(x_(i+1), t_n)) \
  & dots.v \
  & = (u_t + k/2 u_(t t) + ...) - beta (u_(x x) + h^2/12 u_(x x x x) + ...) \
  & = k/2 u_(t t) - beta h^2/12 u^(("iv")) + ...
$

This is consistent because it goes to zero as $k, h -> 0$.

Here we can again a trick. Noticing that $u_t = beta u_(x x)$ so $u_(t t) = beta^2 u_(x x x x)$, we can write the final expression as:

$
  tau & = k/2 u^(("iv")) - beta h^2/12 u^(("iv")) + ... \
      & = (k/2 beta^2 - beta h^2/12) u^(("iv")) + ...
$

so if we choose $k = h^2/(6 beta)$ we get $tau = O(h^4)$.

#faint[== Exercise 2]

We want to solve using BE + FD2.

#let vbar = math.lr($|$, size: 200%)

Backward Euler is
$
  "Backward Euler:" quad & (U^(n+1) - U^n)/k = F(U^(n+1), t_(n+1)) \
  "FD2:" quad & (U_(i-1) - 2U_i + U_(i+1))/h^2 approx (diff^2 u)/(diff x^2)vbar_(x_i) \
  => & (U_i^(n+1) - U_i^n)/k = beta ((U_(i+1)^n - 2U_i^(n+1) + U_(i-1)^(n+1))/h^2)
$

Given

$
  mu = k/h^2 \
  -mu U_(i-1)^(n+1) + (1 + 2mu) U_i^(n+1) - mu U_(i+1)^(n+1) = U_i^n
$

We can write it matricially as:

$
  A arrow(U)^(n+1) = arrow(U)^n \
  A = mat(
    1 + 2mu, -mu;
    -mu, 1 + 2mu, -mu;
    dots.down, dots.down, dots.down;
    space, dots.down, dots.down, -mu;
    space, space, -mu, 1 + 2mu
  )
$

The local truncation error is
$
  tau = -(k/2 + beta h^2/12) u^(("iv")) + O(k^2 + h^4)
$

This is consistent since $tau -> 0$ as $k,h -> 0$. We cannot do the trick to increase accuracy. What's the benefit of BE? That it is implicit, so the eigenvalues are always in the stability region. Specifically, because they are always on the right hand side of the complex plane and the stability region of BE is everything outside the unit circle around $1$. Meanwhile, for FE the stability region is the _inside_ of the disk centered at $-1$ so we need to choose $k$ to put it inside.

#faint[== Exercise 4]

We want to apply Jacobi to $u_(x x) = f$. We want to show that Jacobi with that is equivalent to Forward Euler with $u_t = u_(x x) - f$.

Jacobi is:

$
     & (U_(i-1) - 2U_i + U_(i+1))/h^2 = f_i \
  => & U_i^(n+1) = 1/2 (U_(i-1)^n + U_(i+1)^n) - h^2/2 f_i
$


Forward Euler is:

$
  (U_i^(n+1) - U_i^n)/k = (U_(i-1)^n - 2U_i^n + U_(i+1)^n)/h^2 - f_i \
  U_i^(n+1) = U_i^n + k/h^2 (U_(i-1)^n - 2U_i^n + U_(i+1^n)) - k f_i
$

If $k = h^2/2$, then
$
  U_i^(n+1) & = U_i^n + 1/2 (U_(i-1)^n - 2U_i^n + U_(i+1^n)) - h^2/2 f_i \
            & = 1/2 (U_(i-1)^n + U_(i+1)^n) - h^2/2 f_i
$

Which is what we had before. Therefore, Jacobi and Forward Euler with $k=h^2/2$ is equivalent and they both converge to the same solution.

#faint[== Exercise 7]

$
  (U_j^(n+1) - U_j^n)/k = b(theta delta_(x x)^2 U_j^n + (1 - theta) delta_(x x)^2 U_j^(n+1))
$

Where $delta_(x x)$ is just the FD2 operator. Ok let's spell it out actually:
$
  delta_(x x)^2 U_j^n = (U_(j+1)^n - 2U_j^n + U_(j-1)^n)/h^2
$

We want to figure out stability with Von Neumann analysis. We are going to take $U_j^n = G^n e^(i j phi)$, where $e^(i j phi)$ are the eigengrid functions.

$
  delta_(x x)^2 U_j^n & = 1/h^2 G^n (e^(i (j - 1) phi) - 2 e^(i j phi) + e^(i (j+1) phi)) \
  & = 1/h^2 G^n e^(i j phi) (e^(-i phi) - 2 + e^(i phi)) \
  & dots.v \
  & = -4/h^2 e^(i j phi) G^n (sin^2 phi/2) \
  (G - 1) & = underbrace(-k b 4/h^2 (sin^2 phi/2), lambda) (theta + (1 - theta) G) \
  & = lambda (theta + (1 - theta) G)
$

From this we get that

$
  G = (1 + lambda theta)/(1 - lambda (1 - theta))
$

For stability we need $abs(G) <= 1$.

If we let $z = -lambda$, given that $lambda <= 0$ we get that $z >= 0$ so $max z = (4 b k)/h^2$.

Therefore, we need

$
  abs(G) = abs((1 - z theta)/(1 + z (1 - theta))) <= 1 \
  -(1 + z (1 - theta)) <= 1 - z theta <= 1 + z (1 - theta)
$

From this we have:
+ $1 - z theta <= 1 + z (1 - theta) => 0 <= z$ which is always true.
+ $-1 - z(1 - theta) <= 1 - z theta => z(2 theta - 1) <= 2$

So, if $theta <= 1/2$ it is unconditionally stable. Otherwise, if $theta > 1/2$ it is conditionally stable. Specifically,

$
  z <= 2/(2theta - 1)
$

Given that the maximum possible value of $z$ is $max z = (4 b k)/h^2$ we get

$
  (4b k)/h^2 <= 2/(2theta - 1) \
  => (2 b k)/h^2 <= 1/(2 theta - 1)
$

This is the CFL.

If $theta = 1$ we have Forward Euler. If $theta = 1/2$ Crank-Nicholson. If $theta = 0$ it is Backward Euler.

#faint[Recommendation:
  + Try to do this Von-Neumann analysis for hyperbolic problems to find stability conditions
  + Derive absolute stability regions for Crank-Nicholson.
]

= Finite elements

#faint[== Execrcise 1]

We have points

$
  0 = x_0 < x_1 < x_2 < x_3 = 1
$

We set $x_1 = 1/6$ and $x_2 = 1/2$. We interpolate using $phi_0, phi_1, phi_2$ and $phi_3$. What is $phi_1$? Well, it interpolates linearly from $(0, 0)$ to $(1/6, 1)$ and then to $(1/2, 0)$, so

$
  phi_1(x) = cases(
    6x quad & "for" x in [0, 1/6],
    -1/(1/2 - 1/6) (x - 1/6) + 1 = 3(1/2 - x) quad & "for" x in [1/6, 1/2],
    0 quad & "otherwise"
  )
$

We now want to plot $v$, which is

$
  v(x) = -phi_0 (x) + phi_2(x) + 2 phi_3(x)
$

This is equivalent to interpolating to $-1, 0, 1, 2$ at the points $0, 1/6, 1/2, 1$.  The resulting plot is the following.

#figure(lq.diagram(
  lq.plot(
    (0, 1 / 6, 1 / 2, 1),
    (-1, 0, 1, 2),
  ),
  width: 80%,
  height: 6cm,
))

it also asks about the slope. The slope is constant in each interval:

#figure(lq.diagram(
  lq.plot(
    (0, 1 / 6, 1 / 6, 1 / 2, 1 / 2, 1),
    (6, 6, 2, 2, 1, 1),
  ),
  ylim: (0, 6.5),
  width: 80%,
  height: 5cm,
))


#faint[== Exercise 2]

Now we have $x_1 = 1/3$ and $x_2 = 2/3$. We want to find the Lagrange interpolant $pi f in V_h$ of $f$, for the cases
+ $ f(x) = x^2 + 1 $
+ $ f(x) = cos(pi x) $

#let span = $"span"$

The space $V_h$ is defined as

$ V_h = span { phi_0, phi_1, phi_2, phi_3 } $

$
  pi f = sum_(i=1)^4 f(x_i) phi_i (x)
$

The interpolant is a straight line connecting the points:

#for (f, f-display) in (
  (x => calc.pow(x, 2) + 1, $x^2 + 1$),
  (x => calc.cos(calc.pi * x), $cos(pi x)$),
) {
  figure(
    lq.diagram(
      lq.plot(
        lq.linspace(-0.1, 1),
        lq.linspace(-0.1, 1).map(f),
        mark: none,
        stroke: (dash: "loosely-dashed"),
      ),
      lq.plot(
        (0, 1 / 3, 2 / 3, 1),
        (0, 1 / 3, 2 / 3, 1).map(f),
        stroke: 1.5pt,
      ),
      width: 80%,
      height: 8cm,
    ),
    caption: [Interpolant of $f(x) = #f-display$],
  )
}

because the basis functions $phi_i$ are linear too.

#faint[== Exercise 3]

We have interval $I = [0, 1]$ and function $f(x) = x^2$

*a)* We want to find the linear interpolant $u(x) = c_0 + c_1 x$ that minimixes $norm(f - f)_2^2$. So $u = P_h f(x) in V_h$.

We can find the coefficients we have to calculate the innerproducts:

$
  sum_i (phi_i, phi_j) c_i = (f, phi_j) quad forall j
$

The basis of $V_h$ we have is

$
  { phi_0, phi_1 } = { 1, x }
$

so the innerproduct gives equations

$
  && mat((1, 1), (1, x); (x, 1), (x, x)) mat(c_0; c_1) & = mat((x^2, 1); (x^2, x)) \
  && => mat(1, 1/2; 1/2, 1/3) mat(c_0; c_1) & = mat(1/3; 1/4) \
$

The solutions are

$
  c_0 = -1/6, quad c_1 = 1
$

Therefore,

$
  u = x - 1/6
$

*b)* We now divide $I$ into two subintervals of equal length. So now we are going to have a 2-piece linear function. We again have to solve the innerproducts but now it's a slightly bigger pain in the ass:

$
  b_0 = (f, phi_0) = integral_0^(0.5) x^2 (1 - 2x) dif x = 1/48 \
  b_1 = (f, phi_1) = integral_0^(0.5) x^2 (2x) dif x + integral_0.5^1 x^2 (2 - 2x) dif x = 7/24 \
  b_2 = (f, phi_2) = integral_0.5^1 x^2 (2x - 1) dif x = 17/48 \
$

Then we have $M_(i, j) = (phi_i, phi_j)$ which is always:

$
  M = h/6 mat(2, 1, 0; 1, 4, 1; 0, 1, 2)
$

And then we get the coefficients $arrow(xi)$ by solving

$
  M arrow(xi) = arrow(b)
$

So our solution is $P_h f = xi_0 phi_0 + xi_1 phi_1 + xi_2 phi_2$.

This is going to be similar but not exactly the same as the linear interpolant. The $L^2$ projection does not necessarily interpolate the points.

#faint[== Exercise 5]

We have basis ${ 1, x, (3x^2 - 1)/2 }$ which is an orthogonal basis (because they are Legendre) on the interval $I = [-1, 1]$. We are going to compute and draw the $L^2$ projection.

*a)* For the function $f(x) = 1 + 2x$ we can easily see that it is formed by
$
  f = p_0 + 2p_1
$

*b)* $f(x) = x^3$. This is not an element of the basis so we need to find the coefficients:

$
  c_0 = ((x^3, p_0))/((p_0, p_0)) \
  c_1 = ((x^3, p_1))/((p_1, p_1)) \
  c_2 = ((x^3, p_2))/((p_2, p_2)) \
$

we know because of parity that $c_0$ and $c_2$ are $0$, so we only need to compute $c_1$ which turns out to be $3/5$, so

$ v(x) = 3/5 x approx x^3 $

#faint[== Exercise 6]

We have the problem

$-u'' = 7$ for $x in I = [0, 1]$
with initial conditions $u(0) = 2$ and $u(1) = 3$.

*a)* We are going to take finite element space

$
  V_h = { v in C^0([0, 1]) | v|_I "is linear" med }
$

we are going to do a trick to satisfy the boundary conditions:

$
  u = u_h + u_D
$

where $u_h$ is the solution to the $0$ boundary conditions and $u_D = 2 + x$ to satisfy the boundary conditions. We can do this because the differential equation is linear.

*b)* We are going to write the problem in variational form. This consists in findinfg a $u in V_h$ such that

$
  integral_0^1 u' v' dif x = integral_0^1 7 v dif x quad forall v in V_(h, 0)
$

where $V_(h, 0)$ is the version with the $0$ boundary conditions.

*c)* We are going to take equispaced points $x_1 = 1/3$ and $x_2 = 2/3$.

The way to solve this is similar to before, by solving the system
$
  M arrow(xi) = arrow(F)
$

where $xi_i$ are the coefficients of the basis with the hat functions. For it to satisfy the boundary conditions we already know that $xi_0 = 2$ and $xi_3 = 3$. Then, the unknowns are

$
  k = 1/h mat(2, -1; -1, 2) = 3 mat(2, -1; -1, 2)
$

$k$ is the _stiffness matrix_. The right hand side is given by:

$
  F = integral_0^1 7 phi_i = 7/3
$

So the system to solve is

$
  cases(
    3(2 xi_1 - xi_2) = 7/3 + 3 xi_0,
    3(-xi_1 + 2xi_2) = 7/3 + 3 xi_3
  )
$

And from here it's easy to find the coefficients.

#faint[== Exercise 9]

We have problem

$
  -u'' = f quad "for" x in I = [0, L]
$

with boundary conditions

$
  x(0) = x(L) = 0
$

We want to show that the solution $u in V_0$ minimizes the functional

$
  F(w) = 1/2 integral_I w'^2 dif x - integral_I f w dif x
$

So,

$
  F(w) = 1/2 a(w, w) - L(w)
$

given $w = u + v$ we have

$
  F(u + v) & = 1/2 a(u + v, u + v) - L(u + v) \
  & = 1/2 a(u, u) + a(u, v) + 1/2 a(v, v) - L(u) - L(v) \
  & = (1/2 a(u, u) - L(u)) + underbrace((a (u, v) - L(v)), "FEM (variational solution)") + 1/2 a(u, v)
$

and then

$
  F(u + v) = F(u) + underbrace(1/2 integral (v')^2 dif x, >= 0) >= F(u)
$

= Spectral methods

Let's explain the spectral methods matrix

$
  D_n = cases(
    0 quad & "if" i = j,
    1/2 (-1)^(i + j) cot((x_i - x_i)/h) quad & "if" i != j
  )
$

The reason this matrix appears is because the inteprolant is because if you do a bunch of algebra

$
  p(x) &= h/(2pi) sum_(k=-N/2)^(N/2) e^(i k x) \
  &= h/(2pi) (1/2 sum_(k=-N/2)^(N/2 - 1) e^(i k x) + 1/2 sum_(k=N/2 + 1)^(N/2) e^(i k x)) \
  & dots.v quad #text(fill: gray)[(exercise for the reader)] \
  & = h/(2pi) cos(x/2) sin((N x)/2)/sin(x/2)
$

you get

$
  S_n (x) = sin((pi pi)/h)/((2pi)/h tan(x/2))
$

And from this you get

$
  S'_N (x_j) = cases(
    0 quad & "if" j = 0,
    ... quad & "if" j != 0,
  )
$
