= 3.1

$
  dot.double(x) + 2 epsilon dot(x) + x = 0
$

The characteristic equation is

$ r^2 + 2 epsilon r + 1 = 0 $

This has solutions

$
  r = (-2epsilon plus.minus sqrt(4epsilon^2 - 4))/(2a) = -epsilon plus.minus sqrt(epsilon^2 - 1)
$

Since we assumed that $epsilon$ is small, we can assume $epsilon < 1$. Therefore, the solutions for $r$ are complex:

$ r = -epsilon plus.minus i sqrt(1-epsilon^2) $

and therefore the solution for $x$ is of the form:

$
  x = A e^((-epsilon + i sqrt(1 - epsilon^2))t) + B e^((-epsilon - i sqrt(1 - epsilon^2))t)
$

Using the boundary conditions, we get

$
  cases(
    A e^0 + B e^0 = 0 => A = -B,
    A(-epsilon + i sqrt(1 - epsilon^2)) + B(-epsilon - i sqrt(1 - epsilon^2)) = 1
  )
$

And we get

$
  A(-epsilon + i sqrt(i - epsilon^2)) - A(-epsilon - i sqrt(1 - epsilon^2)) = 1 \
  => 2A i sqrt(1 - epsilon^2) = 1 \
  => A = 1/(2i sqrt(1 - epsilon^2))
$

So the solution is:

$
  x(t, epsilon) & = 1/(2i sqrt(1 - epsilon^2)) e^(epsilon t) (e^(i sqrt(1 - epsilon^2)) - e^(-i sqrt(1 - epsilon^2))) \
  & = 1/(sqrt(1 - epsilon^2)) e^(epsilon t) sin(t sqrt(1 - epsilon^2))
$


= 3.2

We rewrite $x$ as

$ x = x_0 + epsilon x_1 + O(epsilon^2) $

So

$
  dot(x) = dot(x_0) + epsilon dot(x_1) + O(epsilon^2) \
  dot.double(x) = dot.double(x_0) + epsilon dot.double(x_1) + O(epsilon^2)
$

Rewriting the ODE, we get

$
  dot.double(x) + 2 epsilon dot(x) + x & = 0 \
  (dot.double(x_0) + epsilon dot.double(x_1) + O(epsilon^2)) + 2 epsilon (dot(x_0) + epsilon dot(x_1) + O(epsilon^2)) + (x_0 + epsilon x_1 + O(epsilon^2)) & = 0 \
  (dot.double(x_0) + x_0) + epsilon (dot.double(x_1) + 2 dot(x_0) + x_1) + O(epsilon^2) & = 0
$

To make this hold regardless of $epsilon$, we need $dot.double(x_0) + x_0 = 0$ and $dot.double(x_1) + 2 dot(x_0) + x_1 = 0$

We also need to take a look at boundary conditions. $x(0) = 0$ is given by $x_0(0) = x_1(0) = 0$. $dot(x)(0) = 1$ means that $dot(x_0)(0) + epsilon dot(x_1)(0) = 0$ and to not make it depend on $epsilon$ we need $dot(x_0)(0) = 1$ and $dot(x_1)(0) = 0$. Then, we clearly get that the solution of

$
  dot.double(x_0) + x_0 = 0
$

with the afformentioned boundary conditions is just $x_0 = sin(t)$.

For the other term we have that

$
  dot.double(x_1) + 2 cos(t) + x_1 = 0 \
  dot.double(x_1) + x_1 = -2cos(t)
$

To solve this we first solve the homgeneous part. The solution to $dot.double(x_1) + x_1 = 0$ is $x_1 = A cos(t) + B sin(t)$.

TODO: Keep solving

= 3.3

= 3.4

= 3.5

= 3.6
