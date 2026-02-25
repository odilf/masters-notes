#import "../format.typ": *

#show: notes[Optimization]

= Linear Programming

Either maximize or minimize given linear constraints:

$
  c^t x
$

== Dual problem

We can find duals of problems:

1. Minimization becomes maximization.
2. Each constraint becomes a variable.
3. Each variable becomes a constraint.
4. The coefficient vectors $c$ and $b$ get swapped.

We can construct the dual using SOB:

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  table.cell(rowspan: 6)[Constraint],
  table.cell(rowspan: 3)[max],
  [Sensible],
  $<=$,
  [Odd], $=$,
  [Bizarre], $>=$,
  table.cell(rowspan: 3)[min], [Sensible], $>=$,
  [Odd], $=$,
  [Bizarre], $<=$,
  table.cell(rowspan: 3, colspan: 2)[Variable sign], [Sensible], $>=$,
  [Odd], [free],
  [Bizarre], $<=$,
)

=== Interpretation of dual problem

Imagine an adversary that tries to maximize $bold(c)^t bold(x)$ (subject to $A bold(x) <= bold(b)$, $bold(x) >= bold(0)$).

We have an agent that controls the activity levels of each component $x_i$, and each activity makes unit profit $c_i$. The vector $b$ specifies the total availability of the resources and the matrix entries $A_(j i)$ is the amount of resource $j$ consumed by one unit of activity $i$.

The adversary now wants to limit the profit the agent makes. Then, the adversary chooses a nonnegative vector $bold(y)$ such that $bold(c)^t <= bold(y)^t A$. The point is that $bold(y)^t A$ are the _prices_ of each of the resource (that makes every feasible production economically unatractive).

That is, $y$ is the minimum

#todo[This is 100% unclear to me.]

Therefore, in the dual we minimize $bold(y)^t bold(b)$ (the cost of production) such that $bold(y)^t A >= bold(c)^t$.

#example[Classic consumer problem][
  We have two goods with some "utility" function $U$. Therefore, the primar problem is
  $
        max & U(x, y) \
    "s.t. " & p_x x + p_y y = I \
            & x, y >= 0
  $

  where $I$ is the income.

  The Lagrangian of this problem is

  $
    cal(L) (x, y; lambda) = U(x, y) + lambda (p_x x + p_y y - I)
  $

  If $x$ and $y$ is feasible, $p_x x + p_y y - I$ is going to be $0$. The point now is that the Lagrangian problem is unrestricted. $lambda$ is the regularization parameter which encodes how unfeasiable is the original problem, so we want to find the minimum $lambda$. That is,

  $
    p^* = max_x min_(lambda in RR) cal(L) (x, y, lambda)
  $

  #faint[Why is there now no $lambda >= 0$ restriction? Because the problem is now an equality constraint.]

  The point is that $lambda$ is the _shadow price_ of the income constraints. If you spend one more euro, are you going to be much more happy (high $lambda$) or are you going to have the same satisfaction ($lambda =0$)? This is also called the _marginal utility of income_.

  Let $p^*$ be the optimal utility, which depends on the income $I$. If we assume that for each income there is a differentiable solution $x^*(I), y^*(I), lambda^*(I)$, then it must be a saddle point of the lagrangian.

  #todo[Finish writing this, slide 111.]
]

#example[Factory production][
  This is a simpler example for the meaning of the dual.

  Imagine we have two products $x_1$ and $x_2$ which need labor and material to be created. We have some profit $c_1$ for $x_1$ and $c_2$ for $x_2$.

  #todo[Unfinished.]
]

== Two-player zero-sum game

Take RPS. The _payoff matrix_ of a player is

$
  A = mat(
    0, -1, 1;
    1, 0, -1;
    -1, 1, 0;
  )
$

but for the other player it is

$
  B = mat(
    0, 1, -1;
    -1, 0, 1;
    1, -1, 0;
  )
$

Given that $A = -B$, this is a zero-sum game. Every win of a player corresponds to the loss of other players.

Now let's take a strategy where player $A$ plays each of rock-paper-scissors with probabilities $(x_1, x_2, x_3) = bold(x)$ and same with $bold(y)$. The expected payoff using these strategies is

$
  cal(E) & = sum_(i=1)^3 sum_(j=1)^3 PP["chooosing row" i "and column" j] a_(i j) \
         & = bold(x)^t A bold(y)
$

what is the expected payoff for the other player? $-bold(x)^t A bold(y)$, because it is zero-sum.

=== min-max search

We have a curious result (*weak duality*):

$
  max_x min_y a_(i j) <= min_y max_x a_(i j)
$

This is because what counts more is what we do first.

What if we make the weak duality an equality? We get a _pure Nash equilibrium_:

$
  a_(i j)^* = max_x min_y a_(i j) = min_y max_x a_(i j)
$

where $a_(i j)^j$ is also called a saddle or something.

A pure strategy is always bad if the other player can choose a response. But we can have a fixed mixed strategy. Then, the expected payoff of $R$ is $bold(x)^t A bold(y)$, while $C$ is $-bold(x)^t A bold(y)$, so $R$ wants to maximize $bold(x)^t A bold(y)$ and $C$ want to minimize it.

So, $C$ goes second and knows $R$'s strategy $x$, so she can pick $bold(y)$ as

$ min_y bold(x)^t A bold(y) $

and, therefore, $R$ wants to maximize this quantity:

$
  max_x min_y bold(x)^t A bold(y)
$

#theorem[Minmax][
  In a two-player zero-sum game the disadvantage of moving first disappears once players are allowed to use mixed strategies, so

  $
    max_x min_y bold(x)^t A bold(y) =
    min_y max_x bold(x)^t A bold(y) =
    bold(x)^*^t A bold(y)^*
  $

  (this is strong duality)
]

== Simplex method

#todo[Presentation by Jaime and Miró]

== Interior point methods

The motivation is that it improves certain aspects of simplex, including that it has worst-case exponential complexity.

The main idea of IPMs is to turn a constrained optimization problem into a sequence of unconstrainted problems. While simplex methods explore the edge of the simplex, IPMs explore the inside region.

So, the number of iterations in IPMs are approximately constant while for simplex the number of iteration grow linearly, but the step per iteration in simplex is constant and cheap, while in IPMs it's more expensive (since we have to do Cholesky for Newton).

So, for a problem

$
  min c^t x \
  "s.t." quad A x = b quad x >= 0
$

we are going to add a penalty for getting close to the barrier, so we instead minimize

$
  min B(x, mu) = c^t x - mu sum_(i=1)^n ln(x_i)
$

where $mu$ is the barrier parameter.

So, for every $mu$ there is an optimal solution $x^*(mu)$ and as $mu -> 0$, the $mu$-specific optimal approaches the original optimal solution.

Then, the conditions we need (called the KKT conditions) are:
+ Primal feasibility: $A x = b$
+ Dual feasibility: $A^t y + s = c$
+ Perturbed complementarity: $x_i s_i = mu$ (which in Simplex is just $mu = 0$)

And then, to solve this, we just use Newton's, which computes a step direction. Then, we choose a step size ensuring that $x_(k+1) > 0$. How? idk.

=== Dual formulation

#todo[This was poorly explained.]



= Unconstrainted problems

Goal:

$
  min f(x) quad "such that" x in RR^n
$

This problem doesn't make sense if $f$ is linear (which is why we need constraints in linear problems).

This problem is not analytically solvable, so we use iterative methods. I.e.,

$
  x_(k+1) = x_k + t_k d_k
$

where $d_k$ is the descent direction that goes to a local minimum and $t_k > 0$ is the stepsize. We want to choose $t_k$ and $d_k$ such that $f(x_(k+1)) < f(x_k)$.

Remark: for $d_k$ to descent we need that $f'(x_k; d_k) = gradient f(x_k)^T d_k < 0$.

== Necessary conditions

Actually $f(x_(k+1)) < f(x_k)$ is _not enough_. It can happen that the descent asymptotically approaches a value that is not the minimum. >Specifically, for $f(x) = x^2$, $x_0 = 2$ and $t_k = 2^(-k-1)$ we get that $x_k = 1 + 2^(-k)$ so it approaches $1$ instead of $0$. It can also happen for different choices that, for example, $x_k = (-1)^k (1+2^(-k))$ if the step sizes are too large.

Therefore, we need the _First Wolfe Condition (Armijo)_:

$
            f(x_(k+1)) & <= f(x_k) - gamma_k t_k \
            f(x_(k+1)) & <= f(x_k) + beta_1 t_k gradient f(x_k)^T d_k \
  "where" quad gamma_k & = - b_1 gradient f(x_k)^T d_k
$

where the idea is that this rejects step sizes that are too big relative to the improvement.

This only rejects the second counterexample with the too big steps. To reject the smaller step sizes we need the _Second Wolfe Condition (Curvature)_:

$
  gradient f(x_k + t_k d_k)^T d_k >= beta_2 gradient f(x_k)^T d_k
$

which is equivalent to saying that the steps must reduce slope sufficiently:

$
  (gradient f(x_k + t_k d_k)^T d_k)/(gradient f(x_k)^T d_k) <= beta_2
$

These two conditions are independent, but

#theorem[
  If $f$ is bounded from below along the direction and $0 < beta_1 < beta_2 < 1$, then there exists an $t_k > 0$ that verifies both Wolfe conditions.
]


== Stepsize selection

- *Constant stepsize* $t_k = t$: simplest, but may not converge if $t$ is too large or may take too long if $t$ is too small.

  But, if $f$ is Lipschitz continuous with constant $L$, then the method converges if $t in (0, 2/L)$. The Lipschitz constant is defined as

  $
    norm(gradient f(x) - gradient f(y)) <= L norm(x - y)
  $

  and measurees how fast the gradient changes.

- *Exact line search*: We choose $t_k$ to minimize $f$ along the ray $x_k + t d_k$, i.e.,

  $
    t_k = "argmin"_(s >= 0) f(x_k + s d_k)
  $

  Sometimes this can be solved analytically. E.g., for quadratic problems, $t_k = min 1/2 x_k^T A x_k$ with $A$ symmetric. In fact, this problem is important because you can locally approximate almost all functions as quadratics.

  #example[Largest step size for quadratic problem][
    If we have problem $min 1/2 x^T A x$, then

    $
                 f(x) & = 1/2 x^T A x \
        gradient f(x) & = A x \
      gradient^2 f(x) & = A \
    $

    Then we can Taylor in an exact manner:

    $
      f(x_(k+1)) & = f(x_k + t_k d_k) \
      & = f(x_k) + t_k gradient f(x_k)^T d_k + 1/2 t_k^2 gradient^2 f(x_k)^T d_k \
      (dif f(x_(k+1)))/(dif t_k) & = gradient f(x_k)^T d_k + t_k d_k^T gradient^2 f(x_k) = 0 \
      t_k & = - (gradient f(x_k)^T d_k)/(d_k^T gradient^2 f(x_k) d_k) = -(x_k^T A d_k)/(d_k^T A d_k) \
    $
  ]

- *Backtracking line search*: This is an approximation method which calculates a step size on the fly. We choose an $alpha$ and $beta$ and use $t_k = alpha beta^k$. This method is used to satisfy the first Wolfe condition (i.e., prevent large step sizes, not small ones). The idea is as follows:

  - Initialize $t_0$ (often $t_0 = 1$)
  - While $f(x_k + t_k d_k) > f(x_k) + beta_1 alpha gradient f(x_k)^T d_k$ reduce $t_k <- alpha t_k$.
  - Set $x_(k+1) = f(x_k + t_k d_k)$
  - What to do with $t_(k+1)$?

== Gradient descent

Choose $d_k = -gradient f(x_k)$ and choose $t_k$ using line search.

#example[
  $
    f(bold(x)) = min 1/2 (x_1^2 + 10 x_2^2)
  $

  Clearly the optimal point is $x^* = mat(0; 0)$ and the optimal value is $0$, but gradient descent does a bunch of zig-zagging to get to $x^*$. This actually happens in general.
]

#lemma[
  Given a continuously differentiable function $f$ and an unconstrained problem

  $
    min f(x), quad x in RR^n
  $

  the minimizing sequence ${x_k}$ generated by the gradient method with exact line search is perpendicular, in the sense that

  $
    (bold(x)_(k+2) - bold(x)_(k+1))^T (bold(x)_(k+1) - bold(x)_k) = 0
  $
]

#proof[
  We have optimal $t^* = (d^t d)/(d^t A d)$. Also, $x_(k+1) - x_k = -t_k gradient f(x_k)$ and $x_(k+2) - x_(k+1) = -t_(k+1) gradient f(x_(k+1))$.

  Defining,

  $
       g(t) & = f(x_k - t gradient(f(x_k))) \
      g'(t) & = (-gradient f(x_k)) gradient f(x_k - t gradient(f(x_k))) \
    g'(t_k) & = -gradient f(x_k) gradient f(x_(k+1)) \
  $

  But we know that $g'(t_k) = 0$ since we have chosen the optimal $t$, so

  $ gradient f(x_(k+1))^t gradient f(x_k) = 0 $
]

#theorem[Convergence of gradient method][
  Given the problem $ min f(x) quad "where" x in RR^n $

  and $gradient f$ is Lipschitz continuous with constant $L >= 0$, then the gradient method converges monotonically to a minimum. Furthermore, $norm(gradient f(x_k)) -> 0$ as $k -> 0$.

  (this works with constant stepsize, exact line search and backtracking line search).
]

#theorem[Convergence of gradient method for quadratic problems][
  Given the problem

  $ x^T A x quad "where" x in RR^n, A succ 0 $

  the sequence ${ x_k }$ generated by the gradient method with exact line search is such that $ f(x_(k+1)) <= ((M - m)/(M + m))^2 f(x_k) $ where $M = lambda_max (A)$ and $m = lambda_min (A)$.
]

*Remark*: The factor $(M - m)/(M + m) = (kappa(A) - 1)/(kappa(A) + 1)$ with the $kappa(A)$ being the condition number of the matrix. Therefore, the convergence is fast for well-conditioned problems and slow for ill-conditioned ones.

#example[
  $
    min f(x) quad "where" x in RR^n \
    f(x) = 1/2 x^T A x - b^T x
  $

  We can find the minimum analytically quickly:

  $ f(x) = gradient f(x) = A x - b = 0 => A x = b $

  which can be solved numerically in $O(n^3)$ using gaussian elimination.

  Doing it with the gradient method, we start at $x_0$ and go down the gradient. Computing $gradient f(x)$ is $O(n^2)$ and we repeat it some number of iterations $k$. If $k$ is small compared to $n$, it takes less than $O(n^3)$!
]

== Newton's method

Choose descent

$
  d_k = (-gradient^2 f(x_k))^(-1) gradient f(x_k)
$

which works for problems $min f(x)$ where $f(x)$ *is convex*.

The idea is that we can Taylor approximate to quadratic term and minimize in that direction:

$
  hat(f) (x + d) approx f(x) + d^T gradient f(x) + 1/2 d^T gradient^2 f(x) d
$

since $f$ is convex, the Hessian $gradient^2 f(x)$ is positive definite so $hat(f)(x + d)$ has a unique minimizer if $hat(f)' (x+d; d) = 0$ #todo[What do we mean here?],

$
  d = (-gradient^2 f(x))^(-1) gradient f(x)
$

Numerically, we never want to do the inverse of the Hessian, we instead solve the system $gradient^2 f(x) d = -gradient f(x)$.

== Conjugate direction method

This is in between Newton's method and gradient descent. We choose directions such that

$
  d_i A d_j = 0, forall i != j
$

#definition[
  A set of vectors ${ d_1, d_2, ... , d_n }$ is said to be _A-orthogonal_ or _A-conjugate directions_ if $d_i A d_j = 0$ when $i != j$ (and I think nonzero otherwise).
]

#theorem[
  The A-conjugate directions are linearly independent and form a basis of $RR^n$
]

One way to find them is using Gram-Schmidt, but also iteratively which I think we prefer?

The idea is that the minimum $x^*$ that holds at $A x^* = b$ can be written as a linear combination of A-conjugate directions. But, notice that

$
  d_i^T A x^* = d_i^T A (x_1 d_1 + x_2 d_2 + ... + x_n d_n) = x_i d_i^T A d_i
$

Hence,

$
  x_i = (d_i^T A x^*) / (d_i^T A d_i) = (d_i^T b) / (d_i^T A d_i)
$

So if we have the A-conjugate directions we can easily find the minimum.

= Non-linear optimization

== Optimization with equality constraints

#example[
  We want to design a bottle of  given volume $V$ and minimal surface area. Therefore, we want to minimize

  $
    min S(r, h) = w pi r h + 2 pi r^2 \
    "s.t." quad pi r^2 h = 1
  $

  To solve this, we can solve for a variable and then we get an unconstrained problem. Here, if we solve for $h$ we get $h = 1/(pi r^2)$ so then we have the _unconstrained_ problem:

  $
    min S(r, h)|_(pi r^2 h = 1) = 2pi (1/(pi r) + r^2)
  $

  However, it is often not feasible to get analytical solutions.
]

=== Lagrange multipliers

The idea is that the level set curve of a function $f$ is *tangent* to the curve defined by the constraint. That is, if we have problem

$
  max f(x) \
  "s.t." quad g(x) = c
$

at the maximizer $x^*$ we have that

$
  gradient f(x^*) = lambda gradient g(x^*)
$

where $lambda$ is an unknown constant. This effectively converts the constrained problem to an unconstrained problem, by maximizing the Lagrangian multiplier

$
  L(x; lambda) = f(x) - lambda(g(x) - c)
$

since the partial derivatives produce the previous equations. Therefore, a necessary (but not sufficient) condition for a problem like this is:

$
  gradient L(x^*, lambda^*) = 0
$

#example[
  $
    min f(x, y) = x^2 + y^2 \
    "s.t." quad g(x, y) = (y - 1)^3 - x^2 = 0
  $

  in this example the method doesn't work, since the gradient of $g$ is $0$ at $x^*$. In high dimensions this generally doesn't happen so we just ignore it.
]

// #example[
//   We want to find the min or max of $f(x, y) = x^2 + 2y^2$ subjecy yo $x^2 + y^2 = 4$
// ]

=== Algorithms

For the problem

$
  min f(x) \
  "s.t." quad A x = b
$

where $f$ is convex, $A in RR^(p times n)$  with $p < n$ and $x^*$ is finnite (and the equality constraints are linear, implied by being expressed as a matrix mult).

This problem has a unique solution since both the function and the constraints are linear (*remark*: even if the constraints functions are convex but nonlinear the equality constraints need not be convex).

*Strategies*:
- Eliminate affine equality constraints
- For quadratic problems use null-space or QR
- Use Newton's method

==== Eliminate affine equality constraints

Since $A$ is a fat matrix, we can write it as
$
  { x | A x = b } & = x_0 + { x | A x = 0 } \
                  & = x_0 + { N z | z in RR^(n - p)}
$

where $N$ is a basis for the null space of $A$ (so $A N = 0$), and $x_0$ is any feasiable value.

Then, instead of solving the equality constrianed problem, we solve an unconstrained problem via $x^* = x_0 + N z$:

$
  cases(min_x f(x), "s.t." A x = b) & quad <==> quad min_z tilde(f)(x_0 + N z)
$

to get $N$, we use QR factorization of $A^t$, where

$
  A^t = mat(Q_1, Q_2) mat(R; 0)
$

so $Q_2$ forms a basis for the nullspace of $A^t$.

==== KKT-method <sec-kkt>

We can consider the first order conditions:

$
  gradient f(x^*) + A^t lambda^* = 0 \
  A x^* = b
$

This problem is in general nonlinear, but for the quadratic case we can solve it analytically:

#example[Quadratic problem][
  Consider problem $ min f(x) = 1/2 x^t P x + q^t x + r \
  "s.t." quad A x = b $

  where $P > 0$ symmetric and $A in RR^(p times n)$ with $p < n$ is full rank (actually the conditions for $P$ can be given without loss of generality, since we can divide any matrix into a symmetric and antisymmetric part which doesn't affect the gradient #todo[?]).

  The Lagrangian is given by $ L(x; lambda) = f(x) + lambda (A x - b) $

  Then,

  $ gradient f(x) = P x + q $

  $x^*$ is optimal iff there exists a vector $lambda^*$ such that

  $
    cases(gradient f(x^*) + A^t lambda^* = 0, A x^* = b)
    =>
    cases(P x^* + A^t lambda^* = -q, A x^* = b)
  $

  If we put this into a matrix we get the _KKT matrix_:

  $
    mat(P, A^t; A, 0) mat(x^*; lambda^*) = mat(-q; b)
  $

  #theorem[The KKT matrix is nonsingular iff $P$ is p.d. on the nullspace of $A$ ($x^t P x > 0$ for all $x != 0$ such that $A x = 0$).]

  If the KKT matrix is nonsingular, we can just solve this system.
]

We can generalize this to non-quadratic problems...

==== Newton's method

Same as Newton's method for unconstrained problems, except that we have to start at a feasible point and the steps must be feasible. The way we do this is by replacing the problem with a quadratic approximation:

$
  cases(min f(x), "s.t." A x = b) ~~> cases(min_d hat(f)(x + d) = f(x) + d^T gradient f(x) + 1/2 d^T gradient^2 f(x) d, "s.t." A(x + d) = b)
$

then, we can use the previous quadratic method to find the Newton step at $x$:

$
  mat(gradient^2 f(x), A^t; A, 0) mat(Delta x; lambda) = mat(-gradient f; 0)
$

$gradient^2 f(x)$ needs to be p.d., which it is because $f$ is convex.

Funnily enough, if the starting point is unfeasible you can get to a feasible point via

$
  mat(gradient^2 f(x), A^t; A, 0) mat(Delta x; lambda) = -mat(gradient f; A x - b)
$

== Optimization with inequality constraints

we have problem

$
  max f(x) \
  "s.t." quad g(x) <= b
$

Note that in this kind of problem we have no maximum number of constraints.

#definition[Binding and non-binding restrictions][
  A restriction is _binding_ or _active_ at a point $x_0$ if $g_i(x_0) = b_i$. Otherwise, it is _non-binding_ or _non-active_.
]

#definition[Regular point][
  Given that $gradient g_i (x)$ exists, a point $x_0$ is _regular_ if either
  + No restriction is binding at $g_i$
  + All $gradient g_i$ that bind $x$ are linearly independent.
]

The way we solve these is by splitting the cases in two:
+ If the optimum $x^*$ is at the boundary (s.t. $g(x^*) = 0$), then $gradient f(x^*)$ and $gradient g(x^*)$ are parallel, so
  $ gradient f(x^*) = lambda gradient g(x^*) $
  with $lambda > 0$ (if $lambda < 0$, we could move inside the valid region to increase $f$).
+ If the optimum $x^*$ is inside the region (s.t. $g(x^*) < 0$). Then, $gradient f(x^*) = 0$ so the previous equation still holds.

Therefore, the optimality condition is just

$
  gradient f(x^*) = lambda gradient g(x^*) \
  lambda > 0
$

We can write this condition using _complementary slackness_:

$
  L(x; lambda) = f(x) + lambda (b - g(x))
$

and the condition becomes:

$
  gradient_x L(x^*; lambda^*) = 0
$

since that gives:

$
  gradient_x L(x; lambda) = gradient f(x) - lambda gradient g(x) & = 0 \
  gradient f(x) & = lambda gradient g(x)
$

Apparently you can do this with KKT conditions (@sec-kkt), and if $x^*$ is a KKT point of a convex problem, $x^*$ is the global optimum.

Note that if $g$ is concave, $-g$ is convex. So for a convex $f$ and concave $g$ then the problem $min f(x) "s.t." g >= b$ is convex.

#todo[There are barrier methods in the slides, but I think we didn't see them in class.]

= Cut problems

== Min-cut/max flow

This is a problem where in a directed weighted graph, we want to find the maximum flow that can be transported in a graph between a source and a sink node; or equivalently, find the cut with the minimum weight that disconnects the graph. This is because the min cut is the bottleneck of the maximum flow.

The problem is find a flow $f: V times V -> RR_(>= 0)$, such that $f(u, v) <= c(u, v)$ and we maximize $sum f$.

The cut formulation is cutting a graph of $V$ into two disjoint sets $S$ and $T$ where we minimize the sum of the capacities that join $S$ to $T$.

=== Ford Fulkerson

First we choose an _augmenting path_, which is a simple path from source to sink such that we can add a flow. We need that
- $c(u, v) - f(u, v) - gamma >=$ for forward edges
- $f(u, v) - gamma >= 0$ for backward edges (but not $s$ or $t$)

To find the maximum flow you can add is $gamma(u, v) = c(u, v) - f(u, v)$ for forward edges and $f(u, v)$ for the backwards ones, and the minimum of all these is the maximum flow we can add.

If no other augmenting path is found, we stop. This algorithm is guaranteed to terminate (for rational flows) and find the maximum flow. The time complexity is $O(abs(E) dot abs(f^*))$, which is pseudo-polynomial. For a different algorithm (Edmonds-Karp) has complexity $O(V dot E^2)$.

=== LP formulation

As we might have guessed, the max flow and the min cut are _duals_.

In matrix form,

$
  sum_u f_(u v) - sum_w f_(v w)
$

In the dual, capacity constraints become positive variables (say, $d_(u v)$), and the conservation constraints become unbounded variables (say, $z_v$).

The variables then become constraints, so we want to solve

$
  min sum_(u v) c_(u v) d_(u v) \
  "s.t." quad d_(u v) - z_u + z_v >= 0 quad "with" z_s = 1, z_t = 0
$

the interpretation is that $z_v$ creates a partition into source set ($z = 1$) and sink set ($z = 0$). This is because all variables are binary, so the $d_(u v)$ chooses whether we include the edge $u v$ in the cut.

By strong duality, we deduce that the max flow is the same as the minimum cut.

In the dual, the dual variables $lambda_(u v)$ which are generally the shadow prices, it means that if $lambda_(u v) = 0$ it means that the dual constraint is not saturated. That is, $lambda_(u v)$ tells us if we have to "invest" (read, increase flow) in certain edges.

== Max-cut problem

#example[
  Imagine we have a class of people and want to split it to make two groups that are as quiet as possible. We can build a weighted graph where the weight of each edge represent how loud those two people are together. Therefore, if we want to minimize the loudness we want to cut the graph in such a way that we remove the maximum number of loudness. I.e., a max-cut.
]

Inn general, we have a graph $G = (V, E)$ and we want to take a cut $S$ such that the value of the cut

#let cut = "cut"

$
  cut(S) = abs({ (i, j) in E : i in S, j in overline(S) })
$

is maximized.

#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

This is an NP-hard problem.

#example[

  #diagram(
    node-stroke: white,
    edge-stroke: white,
    $
      1 edge() edge("rd") edge("d") & 2 edge("d") \
      3 edge() edge("ru") edge("u") & 4 \
    $,
  )

  Here, any two vertices form the max-cut:

  #diagram(
    node-stroke: white,
    edge-stroke: white,
    $
      1 edge("d") & 2 edge("d") \
      3 edge("u") & 4 \
    $,
  )
]

*Algebraic formulation*: We have variable $x$ where $x_i in {1, -1}$. An edge contributes to the cut if $x_i x_j = -1$ (i.e., they have opposite signs). With this, we can pose it as an optimization problem. Namely,

$
  max sum_(i j in E) (1 - x_i x_j)/2 \
  "s.t." quad x in {1, -1}^n
$

Let's try to get a lower bound for an expected value. If we choose half the edges at random, the expected value of the cut is $1/2 abs(E)$, and the maximum possible value of the cut is $abs(E)$, so the expectation is that we are going to get at least $50%$ of the actual value of the max cut.

Now, let's try to improve this lower bound. We are going to do this via *relaxation*.

=== Relaxation (Goemans-Williamson, 1995)

The nature of the difficulty of this problem is that the variables are  #todo[disconnected (?)], so instead of taking vectors in $x_i in {1, -1}$, we are going to take unit vectors in $v in RR^n$, so that $x_i x_j => v_i^t v_j$. Then, the problem becomes

$
  max sum_(i j in E) (1 - v_i^t v_j)/2 \
  "s.t." quad norm(v_i) = 1
$

We can take a matrix of inner products $X_(i j) = v_i^t v_j$ and transform the problem

$
  sum_(i j in E) (1 - v_i^t v_j)/2 = sum_(i j in E) (1 - X_(i j))/2 = abs(E)/2 - 1/2 sum_(i j in E) X_(i j)
$

and finally, we get the problem

$
  max 1/2 sum_(i j in E) (1 - X_(i j))/2 \
  "s.t." quad X_(i i) = 1 " and " X succ.eq 0
$

This is a _semidefinite linear program_, which is a generalization of linear programming for matrices. The point is that this can be solved in polynomial time.

Now, given a solution of the relaxed problem, how can we go back to the original problem? Well, given the vectors $v_i$ we just choose half of the unit sphere (one way to do this is choose a random vector $r in RR^n$ and then assign $+1$ or $-1$ based on the sign of the innerproduct with each vector).

How good is this solution? For one edge $(i, j)$, what is the probability of this edge being cut? Well, it's $theta_(i j)/pi$, where $theta_(i j)$ is the angle between $v_i$ and $v_j$. The problem minimizes $(1 - v_i^t v_j)/2 = (1 - cos theta_(i j))/2$ and the value where the desired quantity is minimal is $0.878$ (that is, the minimum of $(theta/pi)/((1-cos theta)/2) = 2/pi theta/(1 - cos theta)$).

Therefore, the expected value of the cut is, at least, $87.8%$ of the real max-cut.

= Lagrange duality

We've seen this for linear programming, and it generalizes to general constrained optimization problems in a natural way. Given a problem

$
  min f(x) \
  "s.t." quad f(x) <= 0
$

we can form the Lagrange function

$
  cal(L)(x, lambda) = f_0 (x) + sum_(i=1)^m lambda_i f_i (x); lambda_i >= 0
$

and we can convert the problem to

$
  min_x cal(L)(x, lambda) = min_x f_0 (x) + sum_(i=1)^m lambda_i f_i (x)
$

this also works for supremum:

$
  sup_x cal(L)(x, lambda) = sup_x f_0 (x) + sum_(i=1)^m lambda_i f_i (x)
$

This problem is easier to solve because it's unconstrained, but it now depends on $lambda$ too.

*Remark*: for any feasible $x$,
$
  f_0 (x) >= f_0 (x) + sum_i lambda_i f_i (x).
$

\

Ok, so now let's consider the supremum of the Lagrangian:

$
  sup_(lambda >= 0) cal(L)(x, lambda) = sup_(lambda >= 0) (f_0 (x) + sum_i lambda_i f_i (x))
$

What happens if $x$ is not feasible? At least one of the conditions $f_i (x) <= 0$ is violated, so there is a $j$ such that $f_j (x) > 0$. But since $lambda_j >= 0$, the supremum just takes the $lambda_j -> oo$, so the supremum of the Lagrangian is unbounded!

So, we deduce

$
  sup_lambda cal(L)(x, lambda) = cases(
    f_0 (x) quad & "if" forall i\, f_i (x) <= 0,
    oo quad & "if" exists j\, f_j (x) > 0,
  )
$

As in linear programming, the primal is

$
  p^* = inf_x sup_lambda cal(L)(x, lambda)
$

and the dual is obtained by swapping the operations

$
  d^* = sup_lambda inf_x cal(L)(x, lambda)
$

Remember that the inner operation "matters more", which is from where we can remember _weak duality_:

$
  p^* >= d^*
$

And for the case when $p^* = d^*$, we have _strong duality_.

Strong duality only holds with _Slater's condition_.

#definition[Strictly feasible problem][
  A problem $min f(x)$ subject to $g_i (x) <= 0$ is _strictly feasible_ if there is some point where all inequality constraints are strictly held (i.e., $exists x forall i, g_i (x) < 0$).
]

#theorem[Slater's condition for convex problems][
  Given a convex problem

  $
    min f(x) \
    "s.t." quad g_i (x) <= 0
  $

  where all $f$ and $g_i$s are convex, if the convex problem is strictly feasible, then the problem has _strong duality_.

  #faint[Note: This is a regularity condition.]
]

Note: unless we have some justification like Slater's, we cannot assume in general that strong duality holds.
