#text(size: 36pt, tracking: -0.5mm)[Modeling and Nonlinear Analysis]

#outline()
// #pagebreak(weak: true)

#set heading(numbering: "1.1)")

= Mathematical models

- Model: set of (generally differential) equations that describes a system.

=== Steps to model:
+ Introduce relevant _independent_ and _dependent_ variables.
+ Formulate the model using conservation and constitutive laws (generally using approximations)
+ Solve the model, either
  - exactly (rare)
  - approximate (analitically or numerically)
+ Contrast model predictions with experimental data (falsify the model)

Examples: 3 body problem, traveling waves (Fischer's equation: $diff_t A = k A(1 - A) + D gradient^2 A$),

=== What makes a good model?
- Relatively simple
- Predicitve power
- Consistent with inital assumptions and _robust_ to relaxing some of them
- Mathematically interesting

Generally there is a tradeoff between comlexity and explainability and computability. Mathematical rigor is sacrificed for scientific insight.

== The importance of being non-linear <importance-non-linear>

Linear problems are waaay easier because they can be linearly superpositioned.
- Effects are proportional to causes
- Problems can be broken down in parts and added back up together

=== Predictability

Determinism isn't enough for predictability (stating that it *is* enough is called _strong determinism_)
- Buttefly effect: practical impossibility. Arbitrarily small changes in initial conditions result in arbitrarily large changes in output.
- There is also _inefficiency/irrelevance_: excessive detail irrelevant to global state of the system. Systems that are wasteful.

= Idk

== Example of how to model

We want to understand the equation of motion of throwing a rock in the air. For isntance, what is going to be the maximum height.

We start with the equation of motion of gravity:

$ (d^2x) / (d t^2) = - g R^2 / (x + R)^2 $

This is hard to work with. How can we simplify? $x$ is going to be a lot smaller than $R$, since $x$ will not be more than a few meters but $R$ is like 6000 km.

So, we take $x << R$. Then $ (d^2 x) / (d t^2) = - R^2 / R^2 g = -g $ This is now easy to solve.

$
  (d^2 x) / (d t^2) &= -g \
  (d x) / (d t) &= -g t + c \
  x &= -g t^2 / 2 + c t + x_0
$

What is going to be the maximum height? Just $x_m = v_0 / g$. Then we can understand what is the good maximum velocity.

$
  x_m &<< R \
  v_0 ^ 2 / (2g) &<< R \
  v_0 &<< sqrt(2 g R)
$ 

Note that this is a _heuristic_ approximation. We haven't really justified anything. We don't know how bad it really is or how to correct it. Also note that we have _linearized_ the equation. As we saw in @importance-non-linear, this is a big deal and it might have removed useful insight of what happens in more complex cases, for example.

== Dimensional analysis

Important dimensions:
- $L$: length
- $T$: time
- $M$: mass
- $theta$: temperature
- $I$: current

Dimension of a quantity $v$ is notated as $[v]$. So, $v = x/t$ so $[v] = L T^-1$.

Note that quantities can also be dimensionless, such as $[alpha] = 1$. These quantities don't depend on measuring system.

=== Applied to the previous problem

- $[x_m] = L$
- $[R] = L$
- $[g] = L T^-2$

We can check that $(d^2 x) / (d t^2) = -g R^2 / (R+x)^2$ works out.

$
  L/T^2 = L T ^ -2 L^2/L^2 \
  L T^-2 = L T^-2 quad checkmark
$ 

But we can also solve the problem by using dimensional analysis and seeing what makes sense.

$x_m$ is a function of mass, gravitational acceleration and intial velocity, $f(m, g, v_0)$. Then $[x_m] = [m^a g^b v_0^c]$. If we replace the units of each of the things, we get

$
  [x_m] &= [M^a (L T^(-2))^b (L T^-1)^c] \
        &= [M^a L^b T^(-2b) L^c T^(-c)] \
        &= [L^(b + c) T^(-2b - c) M^a]
$
