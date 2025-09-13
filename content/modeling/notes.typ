#import "../format.typ": *

#show: notes[Modeling and Nonlinear Analysis]

= Mathematical models

- Model: set of (generally differential) equations that describes a system.

=== Steps to model:
+ Introduce relevant _independent_ and _dependent_ variables.
+ Formulate the model using conservation and constitutive laws (generally using approximations)
+ Solve the model, either
  - exactly (rare)
  - approximate (analytically or numerically)
+ Contrast model predictions with experimental data (falsify the model)

Examples: 3 body problem, traveling waves (Fischer's equation: $diff_t A = k A(1 - A) + D gradient^2 A$),

=== What makes a good model?
- Relatively simple
- Predictive power
- Consistent with initial assumptions and _robust_ to relaxing some of them
- Mathematically interesting

Generally there is a tradeoff between complexity and explainability and computability. Mathematical rigor is sacrificed for scientific insight.

== The importance of being non-linear <importance-non-linear>

Linear problems are waaay easier because they can be linearly superpositioned.
- Effects are proportional to causes
- Problems can be broken down in parts and added back up together

== Predictability

Determinism isn't enough for predictability (stating that it *is* enough is called _strong determinism_)
- Buttefly effect: practical impossibility. Arbitrarily small changes in initial conditions result in arbitrarily large changes in output.
- There is also _inefficiency/irrelevance_: excessive detail irrelevant to global state of the system. Systems that are wasteful.

= Main techniques and motivating examples

== Example: Gravity equations on Earth

We want to understand the equation of motion of throwing a rock in the air. For instance, what is going to be the maximum height.

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

== Technique: Dimensional analysis

Important dimensions:
- $L$: length
- $T$: time
- $M$: mass
- $theta$: temperature
- $I$: current

Dimension of a quantity $v$ is notated as $[v]$. So, $v = x/t$ so $[v] = L T^-1$.

Note that quantities can also be dimensionless, such as $[alpha] = 1$. These quantities don't depend on measuring system.

=== Dimensional analysis applied to projectile example

- $[x_m] = L$
- $[R] = L$
- $[g] = L T^-2$

We can check that $(d^2 x) / (d t^2) = -g R^2 / (R+x)^2$ works out.

$
  L/T^2 &= L T ^ -2 L^2/L^2 \
  L T^-2 &= L T^-2 quad checkmark
$ 

But we can also solve the problem by using dimensional analysis and seeing what makes sense.

$x_m$ is a function of mass, gravitational acceleration and initial velocity, $f(m, g, v_0)$. Then $[x_m] = [m^a g^b v_0^c]$. If we replace the units of each of the things, we get

$ [x_m] &= [M^a (L T^(-2))^b (L T^-1)^c] 
      \ &= [M^a L^b T^(-2b) L^c T^(-c)] 
      \ &= [L^(b + c) T^(-2b - c) M^a]
$

But, we know that $x_m$ has units of length. So:

$ [L] = [L^(b + c) T^(-2b - c) M^a]
\ => cases(
  b+c = 1,
  -2b-c = 0,
  a = 0
)
 $

Solving this system we get $a = 0$, $b = -1$, $c = 2$, so we can conclude that $x_m = alpha v_0^2/g$, using just dimensional analysis. We can't totally determine the problem since there can be a dimensionless factor $alpha$ (where $[alpha] = 1$).

== Example: Domino problem

How do dominoes fall? Specifically, how fast does the wave move (not each individual domino). Let's say they have height $h$, width $w$, distance between dominoes $d$ and the wave moves at speed $v$.

#import "@preview/cetz:0.4.2"
#let dominoes = cetz.canvas({
  import cetz.draw: *
  set-style(
    fill: oklch(72.65%, 17.51%, 168.35deg),
    stroke: white
  )

  let h = 2
  let w = 0.4
  let d = 0.8
  let g = 1

  let N = 10

  for i in range(N) {
    let x = i * (w + d)
    let r = calc.min(-(70deg - calc.pow(i,3) * 0.2deg), 0deg)
    group({
      translate(x: x)
      rotate(z: r)
      rect((0, 0), (w, h), stroke: none)

      if i == N - 1 {
        line((-d, 0), (0, 0), mark: (symbol: "|"), name: "d")
        content(
          ("d.start", 50%, "d.end"),
          padding: .1,
          anchor: "north",
          $d$
        )

        line((0, h/2), (w, h/2), mark: (symbol: "|"), name: "w")
        content(
          ("w.start", 50%, "w.end"),
          padding: .1,
          anchor: "north",
          $w$
        )

        line((3*w/2, 0), (3*w/2, h), mark: (symbol: "|"), name: "h")
        content(
          ("h.start", 50%, "h.end"),
          angle: "h.end",
          padding: .1,
          anchor: "north",
          $h$
        )
      }
    })
  }

  translate(y: h * 1.2)
  line(((w + d) * N * 0.8, 0), ((w + d) * N * 0.95 , 0), mark: (start: "|", end: ">"), stroke: 1pt, fill: white, name: "v")
        content(
          ("v.start", 50%, "v.end"),
          angle: "v.end",
          padding: .1,
          anchor: "south",
          $v$
        )

})

#figure(dominoes)

$ v = f(d, h, w, g) = d^a h^b w^c g^d $

$ [v] &= [d^a h^b w^c g^d]
\ L T^-1 &= L^a L^b L^c L^d T^(-2d)
\       &= L^(a+b+c+d) T^(-2d)
$

$ a + b + c + d = 1 $

$ 2d = 1 => d = 1/2 $
$ a + b + c = 1/2 $
$ d = 1/2 - a - c $

$ v = alpha d^a b^(1/2 - a - c) w^c g^(1/2)
  \ = alpha sqrt(h g) underbrace((d/h)^a, pi_1) underbrace((w/h)^c, pi_2) $

  But here we don't really know that $pi_1$ and $pi_2$ get multiplied, since they're both dimensionless it can be an arbitrary function. Therefore, we cannot say anything more than just a function of the ratios of $pi_1$ and $pi_2$.

  $ v = sqrt(h g) f(pi_1, pi_2) $

  Here, we can do measurements or assumptions.

  Firstly, we can assume that $w << h$ so $w / h approx 0$. This amkes $pi_2 approx 0$ so we can simplify $f(pi_1, pi_2) approx g(pi_1) = f(pi_1, 0)$.

  We can also measure the ratio $G(pi_1) = v / sqrt(h g)$. You can do this by doing different measurements by separating the dominos.

  We get a graph where between nearly $0$ and $0.5$ we get a roughly constant value (and this values starts to go down at around $0.7$). #todo[do graph]

  So we can conclude that $v = 1.5 sqrt(h g)$ with $d < h/2$.

  Something interesting about this example is that we can have more than one dimensionless combinations.

== Systematic dimensional analysis

Now let's try to systematize it a bit. We have a general quantity $q$ we want to calculate the model for, where $q$ is (potentially) a function of $n$ other quantities $q = f(p_1, ..., p_n)$.

We take the dimensions:

$ [q] = L^(l_0) T^(t_0) M^(m_0) $
$ [p_j] = L^(l_j) T^(t_j) M^(m_j) $

And combine them:

$ q = alpha P_1^(a_1) P_2^(a_2), ..., P_n^(a_n) $
$ L^(l_0) T^(t_0) M^(m_0) = alpha
\ L^(a_1 l_1 + a_2 l_2 + ... + a_n l_n)
\ T^(a_1 t_1 + a_2 t_2 + ... + a_n t_n)
\ M^(a_1 m_1 + a_2 m_2 + ... + a_n m_n)
$

You get a linear expression.

$ A a = b $

Where

$ A = mat(
  l_1, l_2, ..., l_n;
  t_1, t_2, ..., t_n;
  m_1, m_2, ..., m_n;
)
quad a = mat(a_1; a_2; dots.v; a_n)
quad b = mat(l_0, t_0, m_0)
$

#todo[what]

$ a = a_p + sum_(k=1)^r gamma_k a_k $

${a_1, a_2, ..., a_k }$ forms a kernel for A.

$ q = alpha Q pi_1^(gamma_1) ... pi_r^(gamma_r) $

Where all $pi_r$ are dimensionless.

We have $[Q] = [q]$ and $Q = p_1^(alpha_p_1) p_2^(alpha_p_2) ... p_n^(alpha_p_n)$
and $pi_k = pi_1^(a_k_1) pi_2^(a_k_2) ... pi_n^(a_k_n)$

$ A a_k = 0$

#example[
  For the previous exercise, we had

  $ cases(
  reverse: #true,
    a + b + c + d = 1,
    -2 d = -1
  ) $

  So the dimension matrix $A$ and the other vectors are:

  $ A = mat(1, 1, 1, 1; 0, 0, 0, -2)
  quad arrow(a) = mat(a; b; c; d)
  quad arrow(b) = mat(1; -1) $
]

Something insightful is that with this analysis we get the reason why dimensionless quantities appear. Namely, it's because the dimension matrix $A$ is rectangular (sometimes). It tells you how many you will have because the #todo[]

#theorem[
  Assuming the formula $q = f(p_1, p_2, ..., p_n)$ is dimensionally homogeneous and dimensionally complete, then it is possible to reduce it to one of the form $q = Q F(Pi_1, Pi_2, ..., Pi_r)$ where $Pi_r$ are independent dimensionless producrts of $p-1, p_2, ..., p_n$. The quantity $Q$ is dimensionally product of $p_1, p_2, ..., p_n$ with the same dimensions as $q$. 
]

*Remark*: the domino example is not totally useless, it can also be applied to carbon nanotubes collapse.

== Technique: Solving PDEs with _similarity variables_

#example[
  Let's start with the diffusion equation:

  $ D (diff^2 x)/(diff x^2) = (diff u)/(diff t) 
  \ [u] = M L^-3 $

  We have $[x] = L$, $[t] = T$ so

  
  $ [D (diff^2 x)/(diff x^2)] = [(diff u)/(diff t)]
  \ [D] [u]/[x^2] = [u]/[t]
  \ [D] cancel([u])/[x^2] = cancel([u])/[t]
  \ [D] = L^2 T^(-1)
  $

  We also need boundary conditions:
  - $u(0, t) = u_0$
  - $u(infinity, t) = 0$
  - $u(x, 0) = 0 "for" (x > 0)$.

  We want to figure out $u$. $u$ can be written as a function $u = f(x, t, D, u_0)$.

  $ [u] &= [x^a t^b D^c u_0^d] 
  \ M L^(-3) &= L^a T^b L^(2c) T^(-c) M^d L^(-3d) 
  \ M L^(-3) &= L^(a + 2c - 3d) T^(b - c) M^d
  \ &=> cases(
    a + 2c - 3d = -3,
    b - c = 0,
    d = 1,
  )
  $

  This has solutions $d = 1$ and $b = c = -a/2$.

  So we can reconstruct $u$:

  $ u &= alpha u_0 x^a t^(-a/2) D^(-a/2)
  \   &= a u_0 (x / sqrt(D t))^a $

  Some things we could've guessed. Namely, $u_0$ has the same dimensions as $u$ so the only possible option for us is to either remove $u_0$ or to get $u_0$ multiplied with a constant factor.

  Since $ x / sqrt(D t)$ is dimensionless, we can't really assume anything about it, so really we have a general function of that quantity:

  $ u = u_0 F(eta) quad "where " quad eta = x/sqrt(D t) $

  But we can look at the derivatives, both in $t$...

  $ (diff u)/(diff t) &= u_0 F'(eta) (diff eta)/(diff t)
  \ &= u_0 F'(eta) x/sqrt(D) (-1/(2 t^(3/4))) 
  \ &= u_0 F'(eta) (-1/(2 t) eta)
  $

  ...and in $x$:

  $ (diff u)/(diff x) &= u_0 F'(eta) (diff eta)/(diff x)
  \ &= u_0 (F'(eta)) / sqrt(D t)

  \ (diff^2 u)/(diff x)^2 &= u_0 (F''(eta))/sqrt(D t) (diff eta)/(diff t)
  \ &= u_0 (F''(eta)) / (D t)
  $

  And we substitute them in
  
  $ cancel(D) cancel(u_0) / (cancel(D) cancel(t)) F''(eta) &= - cancel(u_0) / (2 cancel(t)) eta F'(eta)
  \ F''(eta) &= -eta/2 F'(eta) $

  This is now an ODE! We can solve it more easily. First, make a change of variable $G'(eta) = F(eta)$:

  $ G' &= -eta/2 G
  \ G' / G &= -eta/2 
  \ (log G)' &= -eta/2 
  \ log G &= - eta^2 / 4 + gamma 
  \ G(eta) &= alpha e^(eta^2 / 4) = F'(eta) 
  \ F(eta) &= beta + alpha integral_0^eta e^(-y^2 / 4) dif y $


  Now apply boundary conditions:

  - $u_0 F(0) = u_0 -> F(0) = 1$
  - $u_0 F(infinity) = 0 -> F(infinity) = 0 $
  - $F(0) = 1 = beta$

  $ F(infinity) &= 1 + 2alpha integral_0^infinity e^(-y^2 / 4) dif y / 2
  \ &= 1 + 2alpha integral_0^infinity e^(-xi^2) dif xi
  \ &= 1 + alpha sqrt(pi) = 0 $

#let erf = "erf"
#let erfc = "erfc"
  $ F(eta) = 1 - 1/sqrt(pi) integral_0^eta e^(-y^2/4) dif y
  \ = 1 - underbrace(2/sqrt(pi) integral_0^(eta/2) e^(-xi) dif xi, "error function, erf") 
  \ = 1 - erfc(eta / 2) $

  So finally,

  $ u(x, y) = u_0 "efrc"(x / (2 sqrt(D t))) $

  This $eta$ shows up as a "similarity" function.

  The trick is that we reduce two variables to one, transforming the PDE to an ODE. This doesn't always happen though. If we had more dimensionless quantities it wouldn't work because we would still have two variables and it would still be a differential equation.
]

== Natural scales and scaling

It's a good idea to figure out typical lengths in your problem and non-dimensionalize based on those lengths. Doesn't fundamentally change anything.

#example[Logistic growth][
  #import "@preview/lilaq:0.4.0" as lq

  #let t = lq.linspace(0.1, 2)
  #let gamma = 1

  #let L = 100
  #let t0 = 5
  #let k = gamma

  #lq.diagram(
    lq.plot(t, t.map(t => calc.pow(t, gamma * t)), label: "Exponential"),
    lq.plot(t, t.map(t => L / (1 + calc.exp(-k * (t - t0)))), label: "Logistic"),
  )

  The basic equation of how populations grow is as follows $dot(p) / p = r$ so $p(t) = p_0 e^(gamma t)$ (for a population $p$ and reproductive rate $r$).

  But we have to take into account the exhaustion of resources, so we modify the equation as so:

  $ dot(p)/p = r (1 - p/K) $

  Here, $K$ is the carrying capacity of the environment. As you approach the carrying capacity, the growth rate decreases.

  We can rewrite $ dot(p) = r p(1 - p/K)$. This is logistic growth.

  The dimensions in question are $[K] = [p]$, $[r] = T^(-1)$

  We define $1/r = t_c$ and $p_c = K$. These are natural scales for us. We can define new dimensionless quantities that include these natural scales of our problem:

  $ P = p/p_c = p/K $
  $ tau = t / t_c = r t $

  $ (dif p) / (dif t) &= (dif p)/(dif tau) dot (dif tau) / (dif t)
  \ &= r (dif p) / (dif tau) = r k P (1 - P)
  \ (dif P)/(dif tau) &= P(1 - P) $

  With initial condition $P(0) = P_0 / k$, we can solve it:

  $ (dif P) / (P (1 - P)) = dif tau
  \ 1/(P(1-P)) = 1/P + 1/(1 - P)
  \ (1/P + 1/(1 - P)) dif P = dif tau
  \ log P - log (1 - P) = tau + c
  \ P / (1 - P) = A e^tau
  \ P = (A e^tau) / (1 + A e^tau) = 1 / (1 + 1/A e^(-tau)
   $

  We can calculate $A$ based on $alpha$:
  $ P(0) = alpha = 1 / (1 + 1/A) => 1/A = 1/alpha - 1 $

  And now substitute in:
  $ P(tau) = alpha / (alpha + (1 - alpha) e^(-tau)) $

  We can also rewrite it as

  $ P(tau) = (alpha e^tau) / (1 + alpha(e^tau - 1)) $

  One is nice when $tau$ is large, the other when $tau$ is small.
]

#example[Natural units in projectile problem][
  Remember the initial equation:

  $ (diff^2 x) / (diff t^2) = - (g R^2) / (R + x)^2 $

  We want to find the natural scale of this problem. This is not obvious.

  We assume there exist natural scales of space and time, $x = x_c u$ and $t = t_c tau$.

  Let's substitute in using chain rule for $t$, $dif / (dif t) = dif / (dif tau) dot (dif tau) / (dif t)$:

  $ x_c (dif^2 u) / (dif t^2) = - (g R^2) / (R + x)^2
  \ x_c / t_c^2 (dif^2 u) / (dif tau^2) = - (g R^2) / (R + x)^2
  $

  #todo[Where does this come from?]

  $ x_c / t_c (dif u)/(dif tau) (0) = v_0 $

  $ (dif u) / (dif tau) (0) = (t_c v_0) / (x_c) = pi_3 $

  And also 
  $ (x c) / (t_c^2 g) (dif^2 u) / (dif tau^2) = - 1/(1+x_c/R u)^2 $

  From we which get another two dimensionless quantities:

  $ pi_1 = x_c / (t_c^2 g) \ pi_2 = x_c / R $

  Now we can substitute the dimensionless quantities $pi_n$ into the original equation:
  $ pi_2 (dif^2 u)/(dif tau^2) = - 1 / (1 + pi_2 u)^2 $

  However, we haven't chosen still $x_c$ and $t_c$!

  There are rules of thumb to choose these initial natural scales:
  + Choose based on the initial condition. From this we get $x_c$
  + Pick from the simplified problem. In our case we get rid of $pi_1$ so we get $x_c = t_c^2 g$
  + #todo[Somehow you get this. Fix two of these parameters by making them one?] $pi_2 = v_0^2 / (g R) = epsilon$. If we take $epsilon = 0$ we recover the approximation we got in the first part of the example.
]

= One-dimensional dynamical systems
