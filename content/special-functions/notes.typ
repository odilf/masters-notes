#import "../format.typ": *

#show: notes(
  subtitle: [],
)[Special functions and orthogonal polynomials]

#faint[
  Grading:

  - *40%*: Two partial exams (1h)
  - *60%*: Final exam or presentation
]


= Introduction

Special functions are either in the middle or nowhere, or they connect to a lot of places, depending on how charitable you want to be.

We have elementary functions, consisting of polynomials, rationals, trig, transcedentals (exp, log) and any combination of these. The rest of them, which can't be written in a closed form as an elementary function is a _special_ function.

Typical examples of special functions include:

#let erf = "erf"
#let erfc = "erfc"

- The Gamma function
  $ Gamma(z) & = integral_0^oo t^(z-1)e^(-t) dif t, quad && Re(z) > 0 $
- The Beta function.
  $
    Beta(p, q) &= integral_0^1 t^(p - 1) (1- t)^(q - 1) dif t, quad && Re(p), Re(p) > 0
  $
- The error (and complementary error) function
  $ erf(z) = 2/sqrt(pi) integral_0^z e^(-t^2) dif t $
  $ erfc(z) = 2/sqrt(pi) integral_z^oo e^(-t^2) dif t $
- The Zeta function
  $ zeta(s) = sum_(n=1)^oo 1/n^s, quad Re(s) > 1 $

These are the usual representations, but we can also express a lot of them as power series or other forms, which give us other insights. Also, we can analytically continuate to expand the domain of the functions with restricted domain.

= Gamma and Beta functions

== Gamma function

#definition[Gamma function][
  $
    Gamma(z) = integral_0^oo t^(z - 1) e^(-t) dif t
  $
]

The gamma function is generally well known to be the continuous generalization of the factorial, in the sense that...

#lemma[Generalization of factorial][
  $
    Gamma(n + 1) = n!
  $

  for nonnegative integers $n$.
]

#proof[
  $
    Gamma(n + 1) &= integral_0^oo t^n e^(-t) dif t \
    &=^"by parts" [-t^n e^(-t)]_0^oo + n integral_0^oo t^(n - 1) e^(-t) dif t \
    &= 0 + n dot Gamma(n)
  $

  which, given base case $Gamma(1) = integral_0^oo e^(-t) = 1$, gives the recursive definition of $n!$.
]

Let's look at some properties.

#theorem[Properties of Gamma function][
  - $Gamma(z)$ is analytic (holomorphic) for $Re(z) > 0$.
  - $Gamma(z)$ satisfiees the functional identity $Gamma(z + 1) = z Gamma(z)$, and in general
    $
      Gamma(z + n) = (z + n - 1) (z + n - 2) ... (z + 1) z Gamma(z) = (z)_n Gamma(z)
    $

    where $(z)_n$ is the _Pochhammer symbol_.
  - The Gamma function is a _meromorphic_ function in $CC$, with simple poles at $z = -m, m in ZZ^+$ with residue $(-1)^m/m!$
]

We can use the functional identity $Gamma(z + 1) = z Gamma(z)$ to extend the Gamma function to $Re(z) < 0$. That is, using $Gamma(z) = Gamma(z+1)/z$, and then you can move $z$ towards lower real part and define it for $Re(z) > -1$, and then we can move to $Re(z) > -2$ and so on to our heart's content.

Except, that $0, -1, -2$ are not defined, since the integral diverges at $Gamma(0)$, so it also copies the divergence to the negative integers.

So what happens at the negative integers? Take $z = -m$, where $m = 0, 1, 2$ and consider $(z + m) Gamma(z)$:

$
  (z + m) Gamma(z) & = (z+m) Gamma(z+1)/z \
                   & = (z + m) Gamma(z + 2)/(z(z+1)) \
                   & dots.v \
                   & = (z + m) Gamma(z + m + 1)/(z(z+1) ... (z + m))
$

We can take a limit as $z -> -m$:

$
  lim_(z -> -m) (z + m) Gamma(z) &= lim_(z -> -m) cancel((z + m)) Gamma(z + m + 1)/(z (z + 1) ... cancel((z + m))) \
  &= Gamma(1)/((-m)(-m+1)(-m+2) ... (-1)) \
  &= (-1)^m 1/(m(m+1)(m+2) ... 1) \
  &= (-1)^m/m!
$

Generally, if we get infinities when analytically continuing, they might be "fake", but here we see they are real.

#lemma[Alternative Gamma function formula][
  $
    1/Gamma(z) = z e^(gamma z) Pi_(n=1)^oo (1 + z/n) e^(-z/n)
  $

  where $gamma$ is _Euler-Maschedorni's_ constant.
]

This alternative definition is useful because it you can quickly find zeros, which can tell you things. Namely,

$
  Pi_(n=1)^oo (1 + z/n) &= lim_(N -> oo) Pi_(n=1)^oo (1 + z/n) \
  &= lim_(N -> oo) Pi_(n=1)^oo (1 + z/n) \
  &= lim_(N -> oo) (1 + z) (1 + z/2) (1 + z/3) ... (1 + z/N) \
$

which is $0$ at the negative integers, so this tells you that $Gamma$ diverges at negative integers.

#lemma[Another alternative Gamma function][
  $
    1/Gamma(z) = 1/(2 pi i) integral_(cal(L)) s^(-z) e^s dif s, quad z in CC
  $

  where $cal(L)$ goes from $oo$ with $arg s = -pi$ to $oo$ with $arg s = pi$, winding down the origin in the positive direction.

  This formula is nice because it is valid over all of $CC$, but the price is that now the integral goes over $CC$ too.
]

We also have incomplete Gamma functions...

#definition[Incomplete Gamma functions][
  $
    gamma(a, z) & = integral_0^z t^(a - 1) e^(-t) dif t \
    Gamma(a, z) & = integral_z^oo t^(a - 1) e^(-t) dif t \
  $

  Often, in practice, these are normalized:

  $
    P(a, z) = gamma(a, z)/Gamma(a) quad Q(a, z) = Gamma(a, z)/Gamma(a)
  $

  so that $P(a, z) + Q(a, z) = 1$.
]

== Beta function

#definition[Beta function][
  $ Beta(p, q) = integral_0^1 t^(p-1) (1-t)^(q-1) dif t, quad p,q > 0 $
]

#lemma[Beta and Gamma relationship][
  $
    Beta(p, q) = (Gamma(p) Gamma(q))/Gamma(p + q)
  $
]

We can use this to prove...

#lemma[Euler's reflection formula][
  $
    Gamma(z) Gamma(1 - z) = pi/(sin (pi z))
  $
]

#let res = "res"
#proof[idea][
  We use the fact that  $ Gamma(z) Gamma(1 - z) = Beta(z, 1 - z) = integral_0^1 t^(z - 1) (1 - t)^(-z) dif t $

  If we restrict $0 < Re(z) < 1$ and do change of variable $t/(1-t) = u$, we get

  $
    Gamma(z) = Gamma(1 - z) &= integral_0^oo (u/(u+1))^(z-1) (1 - u/(u + 1))^(-z) (u+1-u)/(u+1)^2 dif u \
    & = integral_0^oo u^(z-1)/(u+1)^(-1) 1/(u+1)^2 dif u \
    & = integral_0^oo u^(z-1)/(u + 1) dif u
  $

  Then we can do an integral over a contour $C$ onsisting of two circles of radius $epsilon < 1 < R$ together with the negative real axis from $-R$ to $-e$. Then, Cauchy says that $integral_C t^(z-1)/(1 - t) dif t = 2 pi i med res_(t=1) t^(z-1)/(1-t) = -2 pi i$.

  Then we can write the integral over $C$ as four integrals over each segments, which relate to the original integral. Finally, we can use analytic continuation to extend outside of $0 < Re(z) < 1$.
]


== Riemann zeta functionn

#definition[Riemann zeta function][
  $ zeta(s) = sum_(n=1)^oo 1/n^s quad Re(z) > 1 $
]

We can analytically continue this function to the whole complex numbers too. Doing this is complicated, but we are going to see an overview of it.

First, notice that, by a change of variable,

$
  Gamma(s)/n^s = integral_0^oo e^(-n u) u^(s - 1) dif u
$

and also

$
  sum_(n=1)^oo Gamma(s)/n^2 = Gamma(s) zeta(s)
$

since $sum_(n=1)^oo e^(-n x) = 1/(1 - e^(-x)) - 1 = -1/(e^x - 1)$, we can pass the sum to the integral, to get

$
  Gamma(s) zeta(s) = integral_0^oo x^(s - 1)/(e^x - 1) dif x
$

#faint[technically, passing the sum inside the integral needs more justification.]

Now, this has poles at $e^x - 1 = 0 => e^x = 1 => x = 2 k pi i$. That is, there is a whole array of poles in the imaginary axis.

To actually do the analytic continuation, we need to take the integral
$
  integral_oo^((0^+)) (-x)^s/(e^x - 1) 1/x dif x
$

with $exp(s log(-x))$, and we do a bunch of stuff where I got lost.

#todo[I got lost.]

Finally, we get $ zeta(s) = Gamma(1- s)/(2pi i) integral_oo^((0^+)) (-x)^(s-1)/(e^x - 1) dif x $

#todo[There's something here about poles]

At $s=1$, we have

$
  integral_oo^((0^+)) (dif x)/(e^x - 1) = -2pi i
$

by residue analysis #todo[I think??], and so $lim_(s -> 1) (s - 1) zeta(s) = 1$, so $zeta(s)$ has a simple pole. Therefore,

#theorem[Analytic Riemann zeta function][
  $
    zeta(s) = Gamma(1 - s)/(2pi i) integral_oo^((0^+)) (-x)^(s - 1)/(e^x - 1) dif x
  $

  defines a function analytic in $CC \\ {1}$, with a simple pole at $1$.
]

#todo[Missed a day]

= Separation of variables

Many equations appear after separating variables of the Helmholtz equation:

#definition[The helmholtz equation][
  $ Delta u + k^2 u = 0 $

  where $u = u(x, y, z)$ (i.e., it depends on 3 variables).
]

#example[
  
]
