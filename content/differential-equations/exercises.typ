#import "@preview/lilaq:0.5.0" as lq

#show heading.where(level: 1): set heading(numbering: "1.")
#let faint = it => {
  set text(fill: white.darken(60%))
  strong(it)
}
// #show heading: it => {
//   v(2cm)
//   it
// }

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

=
