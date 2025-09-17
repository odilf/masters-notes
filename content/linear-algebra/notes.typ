#import "../format.typ": *

#show: notes[Applied and Computational Linear Algebra]

= Conditioning, floating point arithmetic, computational complexity, and stability

== Floating point arithmetic

Any machine has finite memory, so it can only represent a
finite subset of $RR$. Therefore, we have to choose which set to use, based on:
- How many numbers can we store
- How large is the distance between numbers
- How does the computer do calculations with those numbers

Generally you use either 8, 16, 32 or 64 bits. For our purposes we generally use 32 bits to have precision but not occupy too much memory (also because you can pack and do SIMD so using 32 bits is often twice as fast since you do two calculations instead of one). 32-bit floating point is also called  single
precision.

There are multiple ways to represent these numbers. You can do unsigned and signed integers, but here we generally use floating point, which encodes binary scientific notation, as a sign, an exponent and a mantissa. This is IEEE754.

Distances between consecutive numbers are almost always equal but as numbers get big they get sparser and as they get small, they’re denser. This makes it so that the relative error is about constant which means that the absolute error grows as numbers do.

== Operations in finite memory

Suppose we can only remember 3 digits. E.g., $pi approx 3.14$ or $100$. How can we do $3.14 + 100$? The result is $103.14$, so we can't store them in 3 digits, so we get that $3.14 + 100 = 103$.

This is what happens with computers and bits. Bigger numbers prevail so when adding a big number to a small one you lose information about the small one.

#example[Problem 1.2][
  We will add a large amount of small numbers to $1$. We do this using a for-loop
  which adds all these small numbers one by one to $1$. We consider the number $0.00001$ and add it
  $10000$ times to $1$.

  If we do the naïve approach, we get a decent error:

  ```py
  x = 1
  for _ in range(10_000):
   x += 0.00001

  error = 1.1 - x # Error is ~1.3e-4
  ```

  The error is in the order of $10^(-4)$ with single precision. The minimum error is $10^(-7)$, so we're wasting 3 orders of magnitude which ain't amazing.
]

How close are floating point numbers to real numbers?

#let fl = $"fl"$

Given a number $x in RR$, a computer will round it to the nearest in the set of floats. We use the notation $fl(x)$ to represent a number $x$ on a computer.

What is the difference between $fl(x)$ and $x$? Well, $fl(x) = x(1 + delta)$ for some offset $delta$ that represent how far off you're from the correct value. This error is bounded by $|delta| <= 1/2 beta^(1-p)$ (since $x$ will be no more than half of the distance between numbers). Given a range $[b^(e-1), b^e]$ where we can represent $beta^p$ numbers, the smallest distance between 2 flaots in an interval is $beta^(e-p)$.

Basically, we have $0.underbrace(0000000 ... 001, p) dot beta^(e + 1) = beta^(e - p)$.

Then, $|x - fl(x)| <= beta^(e-p) / 2 => (|x - fl(x)|)/(|x|) <= beta^(e-p)/(2beta^(e - 1))$

/// Machine precision
#let emach = $epsilon_"mach"$
#definition[
  We define the _unit roundoff_ $u$ or the _machine precision_ $emach$ as the smallest value such that $1 + emach$ is distinguishable from 1.

  The value $|delta|$ will never exceed the machine precision, so $|delta| <= epsilon_("mach") = u$.
]

Let's look at the example in more detail knowing this.

#example[cont.][
  The value $1$ is just represented by $0.100... dot 2^1$ exactly.

  The value of $0.00001$ is represented by an annoying representation. Let's look at them in full:

  #grid(
    columns: (auto, auto),
    inset: 2pt,
    $0.00001_10 =$, $#raw("0.1010 0111 1100 0101 1010 1100") _2 * 2^(-16)$,
    $1_10 =$, $#raw("0.1000 0000 0000 0000 0000 0000") _2 * 2^1$,
  )

  We need to bit shift $0.00001$ by $17$ to align the exponents. A few bits get chopped off this way (represented with strikethrough).

  #grid(
    columns: (auto, auto),
    inset: 2pt,
    $10_10^(-5) =$,
    $#raw("0.0000 0000 0000 0000 0101 0011 1110 0010") #strike(raw("1101 0110")) _2 * 2^1$,

    $1_10 =$, $#raw("0.1000 0000 0000 0000 0000 0000 0000 0000") _2 * 2^1$,
  )

  You can compute what this difference is, and it turns out to be:

  $|(1 + 10^(-5)) - fl(1 + 10^(-5)))| = |1.00001 - 1.0000989437| approx 10^(-7)$

  We repeat this a bunch of times, and that's how we arrive at the error (technically Matlab shows a different result than what we have above, but the point stands).
]

Real quick:

#definition[
  When adding floats, we use $plus.circle$ since they're limited precision.

  That is, $x + y$ becomes $fl(x) plus.circle fl(y)$
]

Generally, we don't care about exact results such as the above because, as we see, it's inaccurate. We care about order of magnitude.

== Floating point operations

=== Computational cost

#definition[
  We measure the computational cost of a procedure in the number of operations, or _flops_, represented by $ast.circle$.

  Let $x, y$ be floating point numbers, then $x ast.circle y = (x ast y)(1 + delta)$ where $|delta| <= emach$.

  Meanwhile, for real numbers $x, y in RR$ then we get $fl(x) ast.circle fl(y) = (x(1 + delta_1) ast y(1 + delta_2))(1 + delta_3)$, again with $|delta_n| <= emach$.
]

#example[Cost of inner product and solving system][
  The inner product of two vectors of two vectors $u,v in RR^NN$ does $n$ multiplications and $n - 1$ sums so the total is $2n - 1$ flops, and we mostly care about asymptotic complexity based on problem size. Here, it's runtime $O(n)$

  Solving a system $A x = b$ we have complexity $O(n^3)$.
]

#example[Problem 1.3][
  Consider the quadratic polynomial $p(x) = a x^2 + b x + c$, with $a = 1$, $b = −742$, and $c = 2$.

  Computing the result by hand we get:

  $
    x_(1,2) & = (-b plus.minus sqrt(b^2 - 4 a c)) / (2a) \
            & = (742 plus.minus sqrt(550564 - 8)) / 2 \
            & = (742 plus.minus sqrt(550558)) / 2 \
            & = (742 plus.minus 741.996) / 2 \
        x_1 & = 1484.00 / 2 \
            & = 742 \
        x_2 & = 0.005 / 2 \
            & = 0.0025
  $

  The problem is $x_2$, because we started with two accurate numbers but since they're very close together, almost all precision is lost. The lesson is that subtracting two very similar numbers gives basically meaningless results.

  FYI, a solution here is to just use $x_1$ which still has most of the precision and rewrite $a x^2 + b x + c = a(x - x_1)(x - x_2)$ and solve for $x_2$ to get $x_2 = c / (a x_1)$.

  *Error analysis*:

  We assume that $a$, $b$, $c$, $b^2$, $2a$, $4a c$ and $b^2 - 4a c$ can be represented exactlty.

  We get

  $
    x_(1,2) &= -b plus.circle sqrt(b^2 - 4a c) div.circle 2a
    \ &= -b plus.circle (overbrace(sqrt(b^2 - 4a c), s) (1 + epsilon_1)) div.circle 2a
    \ &= (-b plus.minus (s (1 + epsilon_1))) (1 + epsilon_2) div.circle 2a
    \ &= ((-b plus.minus (s (1 + epsilon_1))) (1 + epsilon_2)) / (2a)(1 + epsilon_3)
    \ &= (-b plus.minus s + s epsilon_1 - b epsilon_2 plus.minus s epsilon_2 + cancel(s epsilon_1 epsilon_2)) / (2a) (1 + epsilon_3)
  $

  In the final step we can neglect $s epsilon_1 epsilon_2$ because it is clearly smaller than $s epsilon_1$, but we couldn't do that if we didn't have that $s epsilon_1$! You have to spell it out, as in "neglected because $s epsilon_1 epsilon_2 << s epsilon_1$".

  Let's continue:
  $
    \ &approx (-b plus.minus s + s epsilon_1 - b epsilon_2 plus.minus s epsilon_2) / (2a) (1 + epsilon_3)
    \ &approx (-b plus.minus s + s epsilon_1 - b epsilon_2 plus.minus s epsilon_2 -b epsilon_3 plus.minus s epsilon_3 + cancel(s epsilon_1 epsilon_3) - b epsilon_2 plus.minus cancel(s epsilon_2 epsilon_3)) / (2a)
    \ &approx (-b plus.minus s + s epsilon_1 + (-b plus.minus s) epsilon_2 + (-b plus.minus s) epsilon_3) / (2a)
    \ &approx underbracket((-b plus.minus s) / (2a), "exact") + underbracket((s epsilon_1 + (-b plus.minus s)(epsilon_2 + epsilon_3)) / (2a), "error") = tilde(x)_(1,2)
  $

  We can now compute the relative error:

  $
    (|x_(1,2) - tilde(x)_(1,2)|) / (|x_(1,2)|) &= (|s epsilon_1 + (-b plus.minus s)(epsilon_2 epsilon_3)|) / (-b plus.minus s)
    \ &<= (|s epsilon_1|) / (|-b plus.minus s|) + |epsilon_2 + epsilon_3| space "(triangle ineq.)"
    \ &<= (|s emach|) / (|-b plus.minus s|) + 2 emach
    \ &= (|s|) / (|-b plus.minus s|) emach + 2 emach
  $

  So, finally, can calculate that $-b - s tilde O(10^(-3))$ we see that $O(10^3) / O(10^(-3)) emach = O(10^6) emach tilde O(1)$. This means the result is very bad.
]

== Pipeline to solve a mathematical problem

We have the input, or $x$ or "data" and we have a mathematical problem that defines a function (in some sense) that maps the input to the output (or $y$ or "solution"). If this is very hard we do it by approximating it. That's like the point of the course.

We reformulate our mathematical problem to a linear algebra problem. We have a matrix $A$ a vector $b$ and we want to get some property of these. To get the properties we have algorithms. We can do this via direct or iterative methods. We get $tilde(f)(dot)$ which approximates the real function $f(dot)$ we want.

Then $tilde(f)(tilde(x))$ approximates $f(x)$ where $tilde(x)$ refers to perturbed $x$, generally because of float rounding error or measurement error (and actually you may also get slightly disturbed $tilde(A)$ and $tilde(b)$ earlier).

Hopefully we can get $tilde(f)(tilde(x))$ close to $f(x)$, that is, an accurate approximation.

Problems conditioning happens in the reformulation step. High conditioning is bad. It is the sensitivity of $f$ under perturbartions of $x$.

Stability happens in the algorithm. This quantifies how sensitive the algorithm itself is to perturbations.

We estimate the error by analyzing conditioning + stability.

Once we are finished we often wonder *how accurate is $tilde(f)(tilde(x))$*?

We can answer this based on accurateness (how "good" the approximation is) which is the relative error based on some norm:

$ (||tilde(f)(tilde(x)) - f(x)||) / (||f(x)||) $

We can try to analyze the _forward error_. Given a function $f$ we have:

$
  underbrace(tilde(f), "algorithm") = underbrace(f, "exact map") + underbrace(Delta f, "perturbation")
$

We have

$ "input" x overparen(->, "prob.") "output" f(x) $
$ "input" x overparen(->, "alg.") "output" tilde(f)(x) $

The forward error is $norm(Delta f)/norm(f)$. The mapping function $tilde(f)$ deviates a bit from $f$ so $y = f(x)$ is going to be a bit different from $tilde(y) = tilde(f)(x)$. We want the difference $|Delta y| = |y - tilde(y)|$ but this is difficult to estimate directly. Therefore, we estimate it indirectly.

Let's try to do it with the _backward error_. While the forward error sees the modification to the output, here we are going to analyze modification in the input.

So we have $x$, we apply $tilde(f)(x)$ to get $tilde(y)$. Instead of trying to understand what happens for $f$ with this $x$, we are going to examine for which perturbed $tilde(x)$ goes to $tilde(y)$ with the exact $f$.

The error we have is $|Delta x| = |x - tilde(x)|$ such that $tilde(f)(x) = f(tilde(x))$.

#example[
  For which $tilde(x) = x + Delta x$ do we have $tilde(f)(x) = f(tilde(x))$?

  We have $tilde(f)(x) = 1.4142$. For which $tilde(x)$ do we have $f(tilde(x)) = 1.4142$? Well, for $sqrt(tilde(x)) = 1.4142 => tilde(x) = 1.99996164$

  $|Delta x| = |x - tilde(x)| = |2 - 1.99996164| = 0.00003836$
]

We will relate the backward error to the forward error via _conditioning_.

Let's properly define conditioning:

#definition[Conditioning][
  A problem is _well-conditioned_ if small changes in the input $x$ result in small changes in the output $y$.

  A problem is _ill-conditioned_ if small changes in the input $x$ result in large changes to the output $y$.

  This is a property of the _function itself_. It has nothing to do with any algorithm you use to compute it.
]

#definition[Condition number][
  The _condition number_ $kappa$
  quantifies the conditioning of a function $f$ by expressing the rate of change of $y$ with respect to $x$.

  It is defined as

  $
    kappa & = sup "relative change in output"/"relative change in input" \
          & = lim_(delta -> 0) sup_(norm(Delta x) / norm(x) <= delta)
            (norm(f(x + Delta x) - f(x)) div norm(f(x))) /
            ((norm(x + Delta x - x)) div norm(x))
  $

  More concretely in words, it is the maximum ratio of how further $f(x)$ can go away. So, for a region around $x$ how much bigger is the region around $f(x)$, as the size around $x$ goes to zero.

  This is conceptually approximately to the forward error divided by backward error, even though it isn't strictly the case.

  If $f(x)$ is differentiable then $kappa_f (x)$ can be equivalently defined as

  #todo[proper jacobian $J$]
  $ k_f(x) = norm(x) / norm(f) norm(J_f) $

  Where $J$ is the Jacobian.

  Small $kappa$ is well-conditioned (good), large $kappa$ is ill-conditioned (bad).
]

We also have the absolute conditional number which are:

$
  kappa_f^"abs"(x) = lim_(delta -> 0) sup_(norm(Delta x <= delta)) norm(f(x + Delta x) - f(x)) / norm(x + Delta x - x)
$

and for differentiable functions $f$:

$ kappa_f^"abs" = norm(J_f) $

#example[11][
  We have $f : RR \\ { 0 } -> RR : x -> f(x) = 1/x$

  $f$ is differentiable, so we can do the Jacobian.

  $ J_f = (dif f) / (dif x) (x) = -1/x^2 $

  So the absolute condition number is:

  $ kappa_f^"abs" = norm(-1/x^2) = |-1/x^2| = 1/(|x^2|) $

  For $x -> 0$, $kappa_f^"abs" -> infinity$ so $f$ in absolute terms is ill-conditioned.

  For relative errors,

  $ kappa_f = (|x|)/(|f(x)|) dot |1/x^2| = (|x|) / (|1/x|) dot |1/x^2| = 1 $.

  Therefore $f$ is well-conditioned in relative terms.

  To see how it works, for $x = 10^(-6)$ and $tilde(x) = 10^(-6) + 10^(-10)$ we have the absolute error equation:

  $
    (|1/x - 1/tilde(x)) / (|x - tilde(x)|) = 1/(10^(-12) + 10^(-16)) approx 10^12
  $

  which is a large error, but in relative terms:

  $
    (|1/x - 1/tilde(x)| div |1/x|) / (|x - tilde(x)| div |x|) = 10^12 dot 10^(-6) / 10^6 = 1
  $

  which is now not such a large error.
]

#example[13][
  For $ f : RR^2 -> RR : mat(x; y) |-> x - y $

  The absolute conditioning number is

  $
    kappa_infinity^"abs" = norm(J_f)_infinity = norm(mat((diff f)/(diff x), (diff f)/(diff y)))_infinity = norm(mat(1, -1))_infinity = |1| + |-1| = 2
  $

  $ kappa_1^"abs" = norm(J_f)_1 = max(|1|, |-1|) = 1 $

  Depending on the norm you use (this comes up later).

  Here you can tell that if the absolute error is constant, the relative error will explode for very small $x$ (but will decrease for big $x$). This is generally undesirable because you often want to do accurate computations with small numbers.

  We can just compute the relative conditioning number:

  $ kappa_1 = norm(mat(x; y))_1/(|x - y|) norm(J_f)_1 = (|x| + |y|)/(|x - y|) $

  so $kappa_1$ is large when $x - y$ is small.

  This explains why subtracting two similar numbers you lose the precision.
]

#example[14][
  #todo[Fill in]
]

#example[15][
  Just plugs in numbers for 14 #todo[?]
]

#example[16][
  #todo[Fill in]
]

