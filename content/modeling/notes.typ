#import "../format.typ": *

#show: notes[Modeling and Nonlinear Analysis]

*Course organization*:

- Exam: 60%
- Exercises 40% (max 3)

= Mathematical models

- Model: set of (generally differential) equations that describes a system.

== Steps to model

+ Introduce relevant _independent_ and _dependent_ variables.
+ Formulate the model using conservation and constitutive laws (generally using approximations)
+ Solve the model, either
  - exactly (rare)
  - approximate (analytically or numerically)
+ Contrast model predictions with experimental data (falsify the model)

Examples: 3 body problem, traveling waves (Fischer's equation: $partial_t A = k A(1 - A) + D gradient^2 A$),

== What makes a good model?

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
  (d^2 x) / (d t^2) & = -g \
      (d x) / (d t) & = -g t + c \
                  x & = -g t^2 / 2 + c t + x_0
$

What is going to be the maximum height? Just $x_m = v_0 / g$. Then we can understand what is the good maximum velocity.

$
           x_m & << R \
  v_0^2 / (2g) & << R \
           v_0 & << sqrt(2 g R)
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
   L/T^2 & = L T^-2 L^2/L^2 \
  L T^-2 & = L T^-2 quad checkmark
$

But we can also solve the problem by using dimensional analysis and seeing what makes sense.

$x_m$ is a function of mass, gravitational acceleration and initial velocity, $f(m, g, v_0)$. Then $[x_m] = [m^a g^b v_0^c]$. If we replace the units of each of the things, we get

$
  [x_m] & = [M^a (L T^(-2))^b (L T^-1)^c] \
        & = [M^a L^b T^(-2b) L^c T^(-c)] \
        & = [L^(b + c) T^(-2b - c) M^a]
$

But, we know that $x_m$ has units of length. So:

$
  [L] = [L^(b + c) T^(-2b - c) M^a]
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
    fill: oklch(52.65%, 17.51%, 168.35deg),
    stroke: white,
  )

  let h = 2
  let w = 0.4
  let d = 0.8
  let g = 1

  let N = 10

  // ground
  rect((-2 * w, 0), ((w + d) * N, -0.5), fill: black, stroke: none)

  for i in range(N) {
    let x = i * (w + d)
    let r = calc.min(-(70deg - calc.pow(i, 3.05) * 0.2deg), 0deg)
    group({
      translate(x: x)
      rotate(z: r)
      rect((-w, 0), (0, h), stroke: none)

      if i == N - 1 {
        line((-d - w, h / 3), (-w, h / 3), mark: (symbol: "|"), name: "d")
        content(
          ("d.start", 50%, "d.end"),
          padding: .1,
          anchor: "north",
          $d$,
        )

        line((0, -h * .1), (w, -h * .1), mark: (symbol: "|"), name: "w")
        content(
          ("w.start", 50%, "w.end"),
          padding: .1,
          anchor: "north",
          $w$,
        )

        line((3 * w / 2, 0), (3 * w / 2, h), mark: (symbol: "|"), name: "h")
        content(
          ("h.start", 50%, "h.end"),
          angle: "h.end",
          padding: .1,
          anchor: "north",
          $h$,
        )
      }
    })
  }

  translate(y: h * 1.2)
  line(
    ((w + d) * N * 0.5, 0),
    ((w + d) * N * 0.75, 0),
    mark: (start: "|", end: ">"),
    stroke: 1pt,
    fill: white,
    name: "v",
  )
  content(
    ("v.start", 50%, "v.end"),
    angle: "v.end",
    padding: .1,
    anchor: "south",
    $v$,
  )
})

#figure(dominoes, caption: [Dominoes setup])

The speed is going to be some function of these variables. We could put more variables in, like air resistance, frictions and so on but we have to make a value judgment as humans and say that, intuitively, it just ain't gonna matter.

$ v = f(d, h, w, g) = d^a h^b w^c g^d $

We retrieve the units, as usual:

$
     [v] & = [d^a h^b w^c g^d] \
  L T^-1 & = L^a L^b L^c L^d T^(-2d) \
         & = L^(a+b+c+d) T^(-2d)
$

$
  a + b + c + d = 1
  \ 2d = 1 => d = 1/2
  \ a + b + c = 1/2
  \ d = 1/2 - a - c
$

And put them back in:

$
  v = alpha d^a b^(1/2 - a - c) w^c g^(1/2)
  \ = alpha sqrt(h g) underbrace((d/h)^a, pi_1) underbrace((w/h)^c, pi_2)
$

But here we don't really know that $pi_1$ and $pi_2$ get multiplied. The equation above just indicates what powers each thing has to be raised to for the units to work out, but since they're both dimensionless it can be an arbitrary (dimensionless) function. Therefore, we cannot say anything more than just having a function of the ratios of $pi_1$ and $pi_2$.

$ v = sqrt(h g) f(pi_1, pi_2) $

To move on from this point, we can do measurements or assumptions.

Firstly, we can assume that $w << h$ so $w / h approx 0$. This makes $pi_2 approx 0$ so we can simplify $f(pi_1, pi_2) approx G(pi_1) = f(pi_1, 0)$.

We can also measure the ratio $G(pi_1) = v / sqrt(h g)$. You can do this by doing different measurements by separating the dominos.

We get a graph where between nearly $0$ and $0.5$ we get a roughly constant value (and this values starts to go down at around $0.7$).

#figure(lq.diagram(
  xlim: (0, 0.8),
  ylim: (0, 2),
  xlabel: $pi_1$,
  ylabel: $G(pi_1)$,
  {
    lq.plot(
      (0.08, 0.12, 0.23, 0.36, 0.48, 0.72),
      (1.48, 1.49, 1.51, 1.47, 1.48, 1.2),
    )
  },
))

So we can conclude that $v = 1.5 sqrt(h g)$ with $d < h/2$.

Something interesting about this example is that we can have more than one dimensionless combinations.

== Systematic dimensional analysis

Now let's try to systematize the procedure so far. We have a general quantity $q$ we want to calculate the model for, where $q$ is (potentially) a function of $n$ other quantities $q = f(p_1, ..., p_n)$.

We take the dimensions:

$ [q] = L^(l_0) T^(t_0) M^(m_0) $
$ [p_j] = L^(l_j) T^(t_j) M^(m_j) $

And combine them:

$ q = alpha P_1^(a_1) P_2^(a_2), ..., P_n^(a_n) $
$
  L^(l_0) T^(t_0) M^(m_0) = alpha
  \ L^(a_1 l_1 + a_2 l_2 + ... + a_n l_n)
  \ T^(a_1 t_1 + a_2 t_2 + ... + a_n t_n)
  \ M^(a_1 m_1 + a_2 m_2 + ... + a_n m_n)
$

You get a linear expression.

$ A a = b $

Where

$
  A = mat(
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

$A a_k = 0$

#example[
  For the previous exercise, we had

  $
    cases(
      reverse: #true,
      a + b + c + d = 1,
      -2 d = -1
    )
  $

  So the dimension matrix $A$ and the other vectors are:

  $
    A = mat(1, 1, 1, 1; 0, 0, 0, -2)
    quad arrow(a) = mat(a; b; c; d)
    quad arrow(b) = mat(1; -1)
  $
]

Something insightful is that with this analysis we get the reason why dimensionless quantities appear. Namely, it's because the dimension matrix $A$ is rectangular (sometimes). It tells you how many you will have because the #todo[]

#theorem[
  Assuming the formula $q = f(p_1, p_2, ..., p_n)$ is dimensionally homogeneous and dimensionally complete, then it is possible to reduce it to one of the form $q = Q F(Pi_1, Pi_2, ..., Pi_r)$ where $Pi_r$ are independent dimensionless producrts of $p-1, p_2, ..., p_n$. The quantity $Q$ is dimensionally product of $p_1, p_2, ..., p_n$ with the same dimensions as $q$.
]

*Remark*: the domino example is not totally useless, it can also be applied to carbon nanotubes collapse.

== Technique: Solving PDEs with _similarity variables_

#example[
  Let's start with the diffusion equation:

  $
    D (partial^2 x)/(partial x^2) = (partial u)/(partial t)
    \ [u] = M L^-3
  $

  We have $[x] = L$, $[t] = T$ so


  $
    [D (partial^2 x)/(partial x^2)] = [(partial u)/(partial t)]
    \ [D] [u]/[x^2] = [u]/[t]
    \ [D] cancel([u])/[x^2] = cancel([u])/[t]
    \ [D] = L^2 T^(-1)
  $

  We also need boundary conditions:
  - $u(0, t) = u_0$
  - $u(infinity, t) = 0$
  - $u(x, 0) = 0 "for" (x > 0)$.

  We want to figure out $u$. $u$ can be written as a function $u = f(x, t, D, u_0)$.

  $
         [u] & = [x^a t^b D^c u_0^d] \
    M L^(-3) & = L^a T^b L^(2c) T^(-c) M^d L^(-3d) \
    M L^(-3) & = L^(a + 2c - 3d) T^(b - c) M^d \
             & => cases(
                 a + 2c - 3d = -3,
                 b - c = 0,
                 d = 1,
               )
  $

  This has solutions $d = 1$ and $b = c = -a/2$.

  So we can reconstruct $u$:

  $
    u & = alpha u_0 x^a t^(-a/2) D^(-a/2) \
      & = a u_0 (x / sqrt(D t))^a
  $

  Some things we could've guessed. Namely, $u_0$ has the same dimensions as $u$ so the only possible option for us is to either remove $u_0$ or to get $u_0$ multiplied with a constant factor.

  Since $x / sqrt(D t)$ is dimensionless, we can't really assume anything about it, so really we have a general function of that quantity:

  $ u = u_0 F(eta) quad "where " quad eta = x/sqrt(D t) $

  But we can look at the derivatives, both in $t$...

  $
    (partial u)/(partial t) & = u_0 F'(eta) (partial eta)/(partial t) \
                            & = u_0 F'(eta) x/sqrt(D) (-1/(2 t^(3/4))) \
                            & = u_0 F'(eta) (-1/(2 t) eta)
  $

  ...and in $x$:

  $
    (partial u)/(partial x) & = u_0 F'(eta) (partial eta)/(partial x) \
    & = u_0 (F'(eta)) / sqrt(D t) \
    (partial^2 u)/(partial x)^2 & = u_0 (F''(eta))/sqrt(D t) (partial eta)/(partial t) \
    & = u_0 (F''(eta)) / (D t)
  $

  And we substitute them in

  $
    cancel(D) cancel(u_0) / (cancel(D) cancel(t)) F''(eta) &= - cancel(u_0) / (2 cancel(t)) eta F'(eta)
    \ F''(eta) &= -eta/2 F'(eta)
  $

  This is now an ODE! We can solve it more easily. First, make a change of variable $G'(eta) = F(eta)$:

  $
          G' & = -eta/2 G \
      G' / G & = -eta/2 \
    (log G)' & = -eta/2 \
       log G & = - eta^2 / 4 + gamma \
      G(eta) & = alpha e^(eta^2 / 4) = F'(eta) \
      F(eta) & = beta + alpha integral_0^eta e^(-y^2 / 4) dif y
  $


  Now apply boundary conditions:

  - $u_0 F(0) = u_0 -> F(0) = 1$
  - $u_0 F(infinity) = 0 -> F(infinity) = 0$
  - $F(0) = 1 = beta$

  $
    F(infinity) & = 1 + 2alpha integral_0^infinity e^(-y^2 / 4) dif y / 2 \
                & = 1 + 2alpha integral_0^infinity e^(-xi^2) dif xi \
                & = 1 + alpha sqrt(pi) = 0
  $

  #let erf = "erf"
  #let erfc = "erfc"
  $
    F(eta) = 1 - 1/sqrt(pi) integral_0^eta e^(-y^2/4) dif y
    \ = 1 - underbrace(2/sqrt(pi) integral_0^(eta/2) e^(-xi) dif xi, "error function, erf")
    \ = 1 - erfc(eta / 2)
  $

  So finally,

  $ u(x, y) = u_0 "efrc"(x / (2 sqrt(D t))) $

  This $eta$ shows up as a "similarity" function.

  The trick is that we reduce two variables to one, transforming the PDE to an ODE. This doesn't always happen though. If we had more dimensionless quantities it wouldn't work because we would still have two variables and it would still be a differential equation.
]

== Natural scales and scaling

It's a good idea to figure out typical lengths in your problem and non-dimensionalize based on those lengths. Doesn't fundamentally change anything.

#example[Logistic growth][
  The basic equation of how populations grow is as follows $dot(p) / p = r$ so $p(t) = p_0 e^(r t)$ (for a population $p$ and reproductive rate $r$).

  But we have to take into account the exhaustion of resources, so we modify the equation as so:

  $ dot(p)/p = r (1 - p/K) $

  Here, $K$ is the carrying capacity of the environment. As you approach the carrying capacity, the growth rate decreases.

  We can rewrite it as $dot(p) = r p(1 - p/K)$. This is logistic growth.

  #figure(
    {
      let t = lq.linspace(0.1, 12)
      let gamma = 0.5

      let L = 100
      let t0 = 6
      let k = 1

      lq.diagram(
        legend: (position: (100% + .5em, 0%)),
        ylim: (auto, L * 1.1),
        lq.plot(
          t,
          t => calc.pow(t, gamma * (t - 0)),
          label: "Exponential",
        ),
        lq.plot(
          t,
          t => L / (1 + calc.exp(-k * (t - t0))),
          label: "Logistic",
        ),
      )
    },
    caption: [Exponential vs logistic growth],
  )

  The dimensions in question are $[K] = [p]$, $[r] = T^(-1)$

  We define $1/r = t_c$ and $p_c = K$. These are natural scales for us. We can define new dimensionless quantities $P$ and $tau$ that are scaled according to these natural scales:

  $ P = p/p_c = p/K quad tau = t / t_c = r t $

  $P$ can easily be interpreted as percentage relative to the carrying capacity. $tau$ is not so easy to interpret, but it is a natural unit that comes from the reproductive rate.

  We can substitute them in on the original equations:

  $
                            (dif p) / (dif t) & = r p (1 - p/K) \
    (dif p)/(dif tau) dot (dif tau) / (dif t) & = r K P (1 - P) \
                cancel(r) (dif p) / (dif tau) & = cancel(r) K P (1 - P) \
                            (dif P)/(dif tau) & = P(1 - P)
  $

  With initial condition $P(0) = p_0 / k$, we can solve it:

  $
    (dif P) / (P (1 - P)) = dif tau
    \ 1/(P(1-P)) = 1/P + 1/(1 - P)
    \ (1/P + 1/(1 - P)) dif P = dif tau
    \ log P - log (1 - P) = tau + c
    \ P / (1 - P) = A e^tau
    \ P = (A e^tau) / (1 + A e^tau) = 1 / (1 + 1/A e^(-tau))
  $

  We can calculate $A$ based on $P(0) = alpha$:
  $ P(0) = alpha = 1 / (1 + 1/A) => 1/A = 1/alpha - 1 $

  And now substitute in:
  $ P(tau) = alpha / (alpha + (1 - alpha) e^(-tau)) $

  #faint[
    We can also rewrite it as

    $ P(tau) = (alpha e^tau) / (1 + alpha(e^tau - 1)) $

    One is nice when $tau$ is large, the other when $tau$ is small.
  ]
]

#example[Natural units in projectile problem][
  Remember the initial equation:

  $ (partial^2 x) / (partial t^2) = - (g R^2) / (R + x)^2 $

  We want to find the natural scale of this problem. This is not clear as for the previous example. What we have to do instead is just assume there exist natural scales of space and time, $x = x_c u$ and $t = t_c tau$, and work accordingly.

  Let's substitute in using chain rule for $t$, $dif / (dif t) = dif / (dif tau) dot (dif tau) / (dif t) = 1/t_c dif / (dif tau)$:

  $
    x_c (dif^2 u) / (dif t^2) = x_c / t_c^2 (dif^2 u) / (dif tau^2) = - (g R^2) / (R + x)^2
  $

  We can also substitute it in the initial condition where $(dif x)/(dif t) = v_0$:

  $
    x_c / t_c (dif u)/(dif tau) (0) & = v_0 \
            (dif u) / (dif tau) (0) & = (t_c v_0) / (x_c) = pi_3
  $

  We can also rewrite the first equation to get two other dimensionless quantities:

  $ (x_c) / (t_c^2 g) (dif^2 u) / (dif tau^2) = - 1/(1+x_c/R u)^2 $

  From we which get:

  $ pi_1 = x_c / (t_c^2 g) quad pi_2 = x_c / R $

  Now we can substitute the dimensionless quantities $pi_n$ into the original equation:
  $ pi_1 (dif^2 u)/(dif tau^2) = - 1 / (1 + pi_2 u)^2 $

  However, we haven't chosen still $x_c$ and $t_c$!

  There are rules of thumb to choose these initial natural scales:
  + Choose based on the initial condition. From this we get $x_c$ #todo[?]
  + Pick from the simplified problem. In our case we get rid of $pi_1$ so we get $x_c = t_c^2 g$ #todo[??]
  + #todo[Somehow you get this. Fix two of these parameters by making them one?] $pi_2 = v_0^2 / (g R) = epsilon$. If we take $epsilon = 0$ we recover the approximation we got in the first part of the example.
]

= One-dimensional dynamical systems

A dunamical system is a state vector where the derivative is some function that depends on the current state #todo[generally, I think].

$
  cases(
    dot(x)_1 & = f_1(x_1, ..., x_n, t),
    & space dots.v,
    dot(x)_n & = f_n(x_1, ..., x_n, t),
  )
$

$ x = F(x, t) $

An autonomous system is a system where the state and the changes of state depend only on the current state. Sometimes you can have explicit dependence on time, but generally the "laws" of physics and most of non-physics don't change. If you repeat an experiment tomorrow you should get identical results.

Imagine a damped harmonic oscillator:

$ m dot.double(x) + mu dot(x) + k x = a cos(t) $

This can be rewritten in the dynamical system form:

$
  cases(
    x_1 = x,
    x_2 = dot(x) = dot(x)_1,
    x_3 = dot.double(x) = dot(x)_2,
    dots.v,
    x_n = x^((n-1)) = dot(x)_(n - 1)
  )
$

$
  cases(
    dot(x)_1 = x_2,
    dot(x)_2 = -mu/m x_2 - k/m x_1,
    dot(x)_3 = 1
  )
$

This system is linear. Some other systems are not linear.

#let phase-space(f, phase-args: none, ..plots) = lq.diagram(
  // width: 11cm,
  height: 7cm,
  lq.quiver(
    lq.linspace(-4, 4, num: 30),
    lq.linspace(-4, 4, num: 30),
    (x, xp) => {
      let y = f(x)
      let norm = calc.norm(xp, y)
      (xp / norm, y / norm)
    },
    pivot: start,
    color: (x, xp, u, v) => -calc.norm(xp, f(x)),
    map: lq.color.map.plasma,
    scale: 0.3,
    ..phase-args,
  ),
  ..plots,
)

#example[Double pendulum][
  This is a nonlinear dynamical system.

  #figure(
    cetz.canvas(
      length: 2cm,
      {
        import cetz.draw: *
        set-style(
          fill: oklch(52.65%, 17.51%, 168.35deg),
          stroke: white,
        )

        let theta = -90deg + 40deg
        line((0, 0), (0, -2))
        line((0, 0), (2 * calc.cos(theta), 2 * calc.sin(theta)))
        arc(
          (0, 0),
          start: 130deg,
          stop: theta,
          mode: "OPEN",
          fill: none,
          radius: 0.5,
        )
      },
    ),
    caption: [Pendulum diagram],
  )

  The equation is

  $ cancel(m) L dot.double(theta) = -cancel(m) g sin(theta) $

  So we get:

  $ dot.double(theta) = -omega_0^2 sin(theta) quad omega_0^2 = g/L $

  With $x_1 = theta$ we get the dynamical system:

  $
    cases(
      dot(x)_1 = x_2,
      dot(x)_2 = -omega_0^2 sin(x_1),
    )
  $

  A lot of times you do small angle approximation where we do $sin(x_1) = x_1$, from which we get a harmonic oscillator.

  $
    cases(
      dot(x)_1 = x_2,
      dot(x)_2 = -omega_0^2 x_1,
    )
  $

  But here, by linearlizing, we lose interesting information. Namely in this example, when you have very high velocity the pendulum keeps rotating instead of oscillating and we completely miss that movement by linearizing.

  The problem is that the equations for the non-simplified case we get elliptical functions. That is hard. With the system we can extract qualitative information from the system without needing to solve it, because solving it is either impractical or unsolvable.

  What we are going to do is introduce _phase space_, where we our going to plot our state vector $x$. When we plot them, time becomes irrelevant because they're encoded in the derivatives. We only care about trajectories, and the direction of the directories so we add arrows on them.

  #let omega = 2
  #phase-space(theta => -calc.pow(omega, 2) * calc.sin(theta))
]

Let's look at a 1D case. Even with some relatively simple function $dot(x) = f(x) = (dif x)/(dif t)$ with $x(t_0) = x_0$ it's very hard to get solutions:

$
  (dif x)/f(x) = dif t \
  integral^x_x_0 (dif u)/f(x) = t - t_0 = G(x, x_0)
$

#phase-space(calc.sin)

#let r = 1
#let K = 2
#phase-space(N => r * N * (1 - N / K))


Let's look at the equations for the logistic equation. For a point $x^*$
- $x = x^* + u$
- $x(t) = x^* + u(f)$
- $dot(x) = dot(u)$

$ f(x) = cancel(f(x^*)) + f'(x^*)u + 0(u) quad u -> 0 $

We assume $f'(x^*) != 0$

Then $x = f(x)$ at the stable points.

$
  dot(u) = f'(x^*)u + o(u) => cases(
    f'(x^*) > 0 => x^* "unstable",
    f'(x^*) < 0 => x^* "stable"
  )
$

This is _linear stability analysis_.


And what if $f'(x^*) = 0$? Then you need to go to the first term that is non-zero. Here you can get half-stable points. Because you end up with $f(x) = f''(x^*)/2 u^2 + O(u^2)$ if $f''(x) != 0$ so $u approx f''(x^*)/2 u^2$ so both sides are either positive or negative so it is stable at one side and negative in the other. #faint[In general, if the first nonzero term is the $n^"th"$, then if $n$ is odd it is either stable or unstable and if it even then it is always half-stable. ]

#theorem[Piccard's][
  If the function $f(x)$ that defines the problem $dot(x) = f(x)$ with $x(0) = x_0$ has a continouous derivative in an interval of $RR$ that contains $x_0$, then there exists a _unique_ solution $x(t)$ valid in some interval in time $(-t, t)$.
]

Why is this important? Now topology plays a role.

Suppose you have a phase plane. If you have a trajectory you can't have another trajectory crossing another trajectory because then the crossing point would have two solutions (neither a trajectory crossing itself), as long as the conditions are met, of course.

A fixed point is also a trajectory, so no trajectories can cross it. This means that any trajectory that might seem to cross a stable point in fact doesn't and just approaches it asymptotically and then there is a separate trajectory that goes out from the fixed point.

The most important point in a one-dimensional phase plane is that they only have monotonic behaviour, from an infinity or a constant to another infinity or constant. Otherwise, you would have crossing points.

#example[
  $dot(x) = 1 + x^2$

  The solution is $x = tan(t)$ so you get:

  #lq.diagram(
    ylim: (-2, 2),
    lq.plot(
      lq.linspace(-calc.pi / 2, calc.pi / 2, num: 100),
      x => calc.tan(x),
    ),

    lq.vlines(-calc.pi / 2, calc.pi / 2, stroke: red),
  )

  Asymptotes at both sides.
]

#example[Overdamped systems][
  An overdamped system is a system where a particle moves in something like honey.

  The derivative is a function and you can do it in terms of potential.

  $ dot(x) = f(x) = -V'(x) $

  The actual equation is

  $ m dot.double(x) = F(x) - underbrace(mu dot(x), "damping") $

  Where $|mu dot(x)| >> |m dot.double(x)|$ to overdamp it.

  And it terms of the potential $V(x)$:

  $ F = -V'(x) $

  In dynamical systems you have

  $ dot(x) = -V'(x) $

  If you calculate the derivative of the potential along a derivate:

  $ dif/(dif t) V(x(t)) = V'(x) dot(x) = -V'(x)^2 <= 0 $

  So whatever the trajectory it never increases the potential.
]

= Bifurcations.

You want your dynamical systems to be robust. Small changes to initial conditions should not drastically change the results.

== Saddle-node bifurcation

The proptotypical example is $dot(x) = r + x^2$. This has two fixed points when $r < 0$(one positive and one negative), at $r = 0$ you have a half-stable fixed point and at $r > 0$ you have no fixed points.

#phase-space(
  x => -2 + x * x,
  phase-args: (label: $r = -2$),
  lq.scatter(
    (-calc.sqrt(2), calc.sqrt(2)),
    (0, 0),
    size: 10pt,
    label: [Fixed points],
  ),
)
#phase-space(
  x => x * x,
  phase-args: (label: $r=0$),
  lq.scatter((0,), (0,), size: 10pt, label: [Fixed point]),
)
#phase-space(x => 2 + x * x, phase-args: (label: $r=2$))

We can look at this in a _bifurcation diagram_:


#{
  let x = lq.linspace(-3, 0, num: 200)
  lq.diagram(
    xlim: (auto, 2),
    lq.plot(x, r => -calc.sqrt(-r), mark: none, label: "stable"),
    lq.plot(x, r => calc.sqrt(-r), mark: none, label: "unstable"),
  )
}

You find this by doing $r + x^2 = 0$ so the stable point of $x$ is at $x = plus.minus sqrt(-r)$

Let's look at another example:

$ dot(x) = r - x - e^(-x) $

The fixed points are at $r - x - e^(-x) = 0$ but this equation is transcendental. We can just plot $x^*(r)$ and we get $r' = 1 - e^(-x)$ and $r' = 0 => x = 0$.

#lq.diagram(
  lq.plot(lq.linspace(-2, 5), x => x + calc.exp(-x)),
)

Which are the stable branches?

We can see this by analyzing $f(x) = r - x - e^(-x)$ by doing a Taylor expansion, so $f(x) = r - x - (1 - x + x^2/2 + O(x^2)) = r - 1 - x^2/2 + O(x^2)$

We can do a change of variable to make our life easier:

$
  s = r - 1 quad y = x/sqrt(2) quad tau = t/sqrt(2) quad dot(x) = dif / (dif t) sqrt(2) y = sqrt(2) (dif y) / (dif tau) (dif tau) / (dif t) = (dif y) / (dif tau)
$

Therefore we get $(dif y) / (dif tau) = s - y^2$, which is called the _normal form_, since it looks the same as the $dot(x) = r + x^2$ equation.

$
  2(r - 1) - x^2 = 0 \
  x^2 = 2(r - 1)
  x^*_plus.minus = plus.minus sqrt(2(r - 1))
$

$
  f(x) = r - 1 - x^2/2 \
  f'(x) = -x \
  f'(x^*_plus.minus) = plus.minus sqrt(2(r - 1))
$

These last equations #todo[somehow] get which side is unstable and which is stable.

All saddle-node bifurcations follows the normal form of $dot(x) = r plus.minus x^2$.

=== Generalizing saddle-node bifurcations

To put this in more general terms, sadle-node bifurcation happens when we have an expression for a derivative $dot(x) = f(x, r)$, with some values $x^*$ and $r_c$. We can taylor expand $f$ around $x^*, r_c$:

$
  f(x, r) & = underbrace(f(x^*, r_c), 0) \
  & + underbrace((partial f)/(partial x) (x^*, r_c), 0) (x - x^*) \
  & + underbrace((partial f)/(partial r)(x^*, r_c), a) (r - r_c) \
  & + underbrace(1/2 (partial^2 f)/(partial x^2), b) (x^*, r_c) (x - x^*)^2 \
  & + underbrace((partial^2 f)/(partial x partial r)(x^*, r_c), c) (r - r_c)(x - x^*) \
  & = a Delta r + b (Delta x)^2 + c Delta r Delta x \
  & = b (Delta x + c/(2b) Delta r)^2 - c^2/4d (Delta r)^2 + a Delta r
$

We do the following change of variables:

$
  y = sqrt(|b|)(Delta x + c/2b Delta r) quad s = a Delta r - c^2/4b (Delta r)^2 \
  tau = t / sqrt(|b|)
$

And we get

$ dot(y) = sqrt(|b|) dot(x) => dot(x) = (dif y) / (dif tau) $

And the bifurcation is at $ (dif y) / (dif tau) = s + y^2 + ... $

That is, for an arbitrary function of the form $dot(x) = f(x, r)$, if it has a saddle-node bifurcation it will be in this "normal form" of $dot(x) = s plus.minus x^2$ (maybe with different variables).

== Transcritical bifurcation

This happens when you have the same fixed point that changes between stable and unstable depending on the parameter. For $r x - x^2$ we always have $0$ as a stable point but it is stable for $r < 0$, unstable for $r > 0$ and half-stable for $r = 0$.

Unsurprisingly, $r x - x^2$ is the normal form of the transcritical bifurcation.

#for (r, rmath) in ((-2, $-2$), (0, $$), (2, $2$)) {
  figure(
    phase-space(x => r * x - x * x),
    caption: [Phase space of $dot(x) = #rmath x - x^2$],
  )
}

This bifurcation can be seen in solid-state lasers. A very simple effective model of a laser is the following. We define $n$ as the number of photons at a time $t$, and we have $N$ the number of excited atoms, and we have a "gain coefficient" $G$. Finally, there is a "loss proportion" $k$ which has units of inverse time, which represents the average lifetime of a photon in the cavity. The relevant equation is:

$ dot(n) = G n N - k n $

We also need an equation of the number of excited atoms $N$ based on the number of photons, which is:

$ N = N_0 - alpha n $

Putting it all together, we get

$
  dot(n) & = G n(N_0) - k n \
         & = (G N_0 - k) n - alpha G
$

This is the normal form of the transcritical bifurcation, so we have that at $N_0 < k/G$ the stable point is $n^* = 0$ and for $N_0 > k/G$ the stable point is $n^* = G(N_0 - k) / (G alpha)$. This shows that below a certain threshold, you have no laser emission, but after the threshold the emissions can grow.

== Pitchfork bifurcation

This is one of the most interesting and typical bifurcations. We get it from the normal form $dot(x) = r x - x^3$. We generally see these bifurcations from systems that have symmetry (we can see that the normal form the change of variable $x -> -x$ leaves it unchanged).


#for (r, rmath) in ((-2, $-2$), (0, $$), (2, $2$)) {
  figure(
    phase-space(x => r * x - x * x * x),
    caption: [Phase space of $dot(x) = #rmath x - x^2$],
  )
}

This bifurcation has always fixed points at $x^* = 0$, but for $r <= 0$ it is stable and otherwise it is unstable. The interesting point is that at $r > 0$ we suddenly get two stable points, at symmetric points since the equation is symmetric.

#{
  let mx = lq.linspace(-3, 0, num: 2)
  let x = lq.linspace(0, 3, num: 25)
  lq.diagram(
    width: 10cm,
    height: 5cm,
    lq.plot(mx, x => 0, label: [stable]),
    lq.plot(x, x => 0, label: [unstable]),
    lq.plot(x, x => calc.pow(x, 1 / 3), label: [stable]),
    lq.plot(x, x => -calc.pow(x, 1 / 3), label: [stable]),
  )
}

Once you cross the zero point, we say that the system breaks symmetry because it has to choose one of the two paths. This is what happens with the Higgs Boson! In physics it's called a _second order phase transition_. We also see it in a 2D icing system.

The zero point is a _supercritical pitchfork bifurcation_. We call it _supercritical_ because the bifurcation happens _after_ the critical point itself.

=== Potential for pitchfork bifurcation

$ dot(x) = r x - x^3 = - (dif V) / (dif x) => V(x) = -r x^2/2 + x^4/4 $

#{
  let x = lq.linspace(-1.5, 1.5, num: 20)
  let v(x, r) = -r * calc.pow(x, 2) / 2 + calc.pow(x, 4) / 4
  lq.diagram(
    lq.plot(x, x => v(x, -1), label: $r=-1$),
    lq.plot(x, x => v(x, 0), label: $r=0$),
    lq.plot(x, x => v(x, 1), label: $r=1$),
  )
}

The one for positive $r$ is sometimes called a double well potential in physics #todo[i think].

#example[
  $ dot(x) = r tanh(x) - x $

  The bifurcation happens at

  $ r = x / tanh(x) $

  Remember the definition of $tanh$:

  $ tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1) $

  #lq.diagram(
    lq.plot(lq.linspace(-5, 5), calc.tanh),
  )

  So the stable points are:

  #lq.diagram(
    lq.plot(lq.linspace(-5, 5), x => x / calc.tanh(x)),
  )

  We can do it analytically by first Taylor expanding:

  $ tanh(x) = x - x^3/3 + O(x^4) $

  So the initial equation becomes:

  $
    dot(x) = r x - r/3 x^3 - x + ... \
    = (r - 1) x - r/3 x^3 + ...
  $

  So the bifurcation is when $r - 1 - r/3 x^2 = 0$, so $ x^* = plus.minus sqrt((r - 1)/(3r)) $
]

== Subcritical pitchfork

Normal form has $dot(x) = r x + x^3$, which is the same as the transcritical but positive instead of negative.

// #for (r, rmath) in ((-2, $-2$), (0, $$), (2, $2$)) {
//   figure(
//     phase-space(x => r * x + calc.pow(x, 3)),
//     caption: $r=#r$
//   )
// }

We can analyze it through the potential, $V(x) = - r x^2/2 - x^4/4$. This looks like this:

#for (r, rmath) in ((-2, $-2$), (0, $$), (2, $2$)) {
  figure(
    lq.diagram(
      lq.plot(
        lq.linspace(-1, 1),
        x => -r * calc.pow(x, 2) / 2 - calc.pow(x, 4) / 4,
      ),
    ),
    caption: [$-#rmath x^2/2 - x^4/4$],
  )
}

This is the opposite as the supercritical pitchfork, because the $x^3$ term is now #emph[de]stabailizing. This becomes unstable after some threshold and goes to infinity, so this system is _unphysical_.

To correct this system we need a negative correction term, so we get

$ dot(x) = r x + x^3 - x^5 $

The minimum term we can introduce is $x^5$ because if we were to put a $x^4$ term, the system would no longer be symmetrical. We could also have a constant in front of $x^5$ but it turns out we can redefine our variables to convert it back to this normal form.

The stable points of the new formula are $x^* = 0$ and the solutions of $r =-x^2 + x^4$.

#lq.diagram(
  xlabel: $x$,
  ylabel: $r$,
  lq.plot(
    lq.linspace(-1.2, 1.2),
    x => -calc.pow(x, 2) + calc.pow(x, 4),
  ),
)

We want to find the minima, so we get $r' = -2x = 4x^3 = 0 => x = 0 "or" x = plus.minus 1/sqrt(2)$.

#todo[TODO: do diagram]
// #{
//   let r = lq.linspace(-3, 3, num: 20)
//   lq.diagram(
//     ylabel: $x^*$,
//     xlabel: $r$,
//     lq.plot((-2, 2), (0, 0)),
//     lq.plot(r, r.map(r => -1/calc.sqrt(calc.abs(r))))
//   )
// }

In physics, this is a _discontinious phase transition_ (sometimes also called 1st order transition). This happens in liquid to gas transition, where the density of the system from gas to liquid has a discontinious jump.


== Imperfect bifurcation

$ dot(x) = h + r x - x^3 $

This system is not exactly symmetrical because you have to negate $h$ as well as $x$.

The fixed points are at $r x - x^3 = -h$.

#figure({
  let x = lq.linspace(-2, 2)
  lq.diagram(
    lq.plot(x, x => 2 * x - calc.pow(x, 3)),
    lq.hlines(1.8, stroke: gray, label: $-h$),
  )
})

#todo[I totally missed this]

In the region between $h = 2(r/3)^(3/2)$ and $h = -2(r/3)^(3/2)$ we have three solutions, otherwise we have only one.

This is kind of hard to analyze.

One idea is to first fix the imperfection parameter $h$. To see what happens we can change variables.

We have $h + r x - x^3 = 0$. We take $r = 3 (h/2)^(2/3) rho$ and $x^* = (h/2)^(1/3) xi$. We end up with:

$
  h + 3 (h/2)^(2/3) rho (h/2)^(1/3) xi - (h/2) xi^3 = 0 \
  h + 3/2 h rho xi - h/2 xi^3 = 0 \
  2 + 3rho xi - xi^3 = 0
$


If instead we fix $r$, we can do a different change of variables, $h = 2(r/3)^(3/2) gamma$ and $x^* = (r/3)^(1/2) eta$. The equation we end up is $2 gamma + 3 eta - eta^3 = 0$

=== Example: Bead on a rotating wire

We have a bead in a wire where that wire is rotating in a vertical axis.

#todo[Make picture?]

The equations we get are

$ m r dot.double(phi) = -m g sin phi + m rho omega^2 cos theta - b r phi' $

For the time being, we assume $b = 0$.

$ rho = r sin phi $

$
  dot.double(phi) = -g/gamma sin phi + omega^2 sin phi cos phi = g/gamma sin phi (gamma cos phi - 1)
$

$ gamma = (omega^2 r) / g $

The equlibrium points are at $phi^* = 0$,$phi^* = pi$ and $cos phi = 1/gamma$ if $gamma >= 1$

We can do taylor expansion of $sin phi (gamma cos phi - 1)$:

$
  sin phi (gamma cos phi - 1)
  & = gamma/2 sin (2phi) - sin phi \
  & = gamma/2 (2phi - 4/3 phi^3) - phi + phi^3/6 + O(phi^4) \
  & = (gamma - 1) phi + (1 - 4gamma)/6 phi^3 + O(phi^4)
$

We get this equation eventually:

$ dot.double(phi) = mu phi + nu phi^3 $

With $mu = g/r(gamma - 1)$ and $nu = (1-4gamma)/6 dot g/gamma$

This is an important equation. It is _Duffing's oscillator_. It is a non-linear "spring" so to say. This equation has $phi$/$-phi$ symmetry, just like the original.
We want to study the _overamped_ limit. We have this equation.

$
  r dot.double(phi) + (b r)/(m g) dot(phi) = -sin phi + gamma sin phi cos phi = sin phi (gamma cos phi - 1)
$

We can introduce _scales of time_. For instance, $t = T tau$ so that $tau$ is dimensionless. The equation above becomes:

$
  r dot.double(phi) + (b r)/(m g) dot(phi) = r/(g T^2) (dif^2 phi) / (dif tau^2) + (b r) / (m g T) (dif phi) / (dif tau)
$

Again, we are interested in the overdamped limit, so it makes sense to make the term that we want to be primary to $1$, that is:

$ (b r) / (m g T) = 1 => T = (b r) / (m g) $

And then, using our new expression for $T$ in the other term:

$ epsilon = (r m^2 g^2) / (g b^2 r^2) = (m^2 g) / (b^2 r) $

So that:

$
  r dot.double(phi) + (b r)/(m g) dot(phi) = epsilon (dif^2 phi) / (dif tau^2) + (dif phi) / (dif tau)
$

To get the overdamped limit, we take $epsilon = 0$. Then, we have:

$ (dif phi) / (dif tau) = sin phi (gamma cos phi - 1) = f(phi) $


#for gamma_val in (0.3, 3) {
  figure(
    phase-space(phi => calc.sin(phi) * (gamma_val * calc.cos(phi) - 1)),
    caption: [$(dif phi) / (dif tau) = sin(phi) (gamma cos(phi) - 1)$, $gamma = #gamma_val$],
  )
}

#let bifurcation-diagram(f) = [];

#bifurcation-diagram((phi, gamma) => (
  calc.sin(phi) * (gamma * calc.cos(phi) - 1)
))

A problem with what we have done is that we have moved from a first order diferential equation to a second order diferential equation. Namely, a second order ODE needs an extra boundary condition than a first order ODE. Here, we can just calculate the derivative at $tau=0$ which is unlikely to be the one measured.

To resolve this, we need to introduce a _singular perturbation_. This is very interesting but we won't get into it in too much detail.

Applied to our problem, the singular perturbation is obtained by getting adding some force. We first introduce a new variable, $Omega = (dif phi) / (dif tau)$. Then, we get

$ epsilon (dif Omega)/(dif tau) + (dif phi)/(dif tau) = F(phi) $

The equations we have now are
$
  cases(
    (dif phi) / (dif tau) = Omega,
    (dif Omega) / (dif tau) = 1/epsilon (f(phi) - Omega)
  )
$

=== Example: Budworm outbreak

This is similar to the logistic function.

$ dot(N) = R N (1 - N/K) - p(N) $

Where $p(N) = -(B N^2)/(A^2 + N^2)$

Let's solve it.

We take $x = N/A$ so $N = A x$ and then

$ A dot(x) = R A x (1- A/K x) - B x^2/(x+x^2) $

Then

$ A/B dot(x) = (R A) / B x (1 - A/K x) - x^2/(1 + x^2) $

Rescale time:

$ t = T tau => A/b dot(x) = A/(B T) (dif x)/(dif tau) $

$ r = (R A)/B, k = K/A $

And we get the equation:

$ (dif x)/(dif tau) = r x (1 - x/k) - x^2/(1 + x^2) $

The fixed points of this equation are $x^* = 0$ and $r(1-x/k) = x/(1 + x^2)$. This is nice because we have a) reduced the number of parameters and b) instead of having to check a very hard elliptic curve, we only need to check a line with $x^2/(1+x^2)$

= Higher dimensional dynamical systems. Multiple species.

In one dimension we only have monotonic behavior. In two, we can have oscillations. In three, we introduce _chaos_.

== Example: Predator-prey systems (Lotka-Volterra)

$ dot(R) = a R - b R L $
$ dot(L) = b R L - c L $

Where $R$ represens rabbits and $L$, lynxes.

This produces waves of rabbits and lynxes, since when you get more lynxes they eat the rabbits so there are less rabbits but then the lynxes have less food so they die so the rabbits reproduce but the lynxes now have more food so you get more lynxes which eat the rabbits... and so on.


Some oscillations are resiliant, some are non-resilliant. In non-resiliant oscillations, a perturbation changes the orbit, while in resilliant systems, the trajectory gets recovered. Resillinat oscillations are also called _self-sustained_.
== Example: SIR model (Kermack-McKendrick)

Model for epidemics or pandemics (Elena's extended essay!).

$ S(t) + I(t) + R(t) = N space ("constant") $

$
  dot(S) & = -beta (S I)/N \
  dot(I) & = beta (S I)/N - gamma I \
  dot(R) & = gamma I
$

=== Motivation: Reactions in chemistry

Take a reaction $A + B -> C$. The rates of change are $dot(A) = dot(B) = -dot(C)$. So we are going to say we have a rate of reaction $r > 0$ so that $dot(A) = -r$, $dot(B) = -r$ and $dot(C) = r$.

What is $r$? It clearly depends on the reactants, so $r = r(A, B)$. We also know that if one of the reactants is missing we won't get a reactiong, so $r(0, B) = r(A, 0) = 0$. And then we also have that any derivative of $r$ when one of the reactants is $0$ will be $0$, that is, $(dif^n r)/(dif A^2) (A, 0) = 0$, $(dif^n r)/(dif B^2) (B, 0) = 0$.

We will now derive the _law of mass action_. We assume that the functions of $A$ and $B$ are analytic, which is a reasonable assumption, and that the concentrations can't be too high, because we're going to take the first terms of a Taylor series. So:

$
  r(A, B) = r_(0, 0) + r_(1, 0) A + r_(0, 1) B + r(1,1) A B + r_(2, 0) A^2 + r_(0, 2) B^2 + O(A^3 + B^3)
$

Let's find the coefficients:

- $r_(0, 0) = r(0, 0) = 0$
- $r_(1, 0) = (dif r)/(dif A) (0, 0) = 0$
- $r_(0, 1) = (dif r)/(dif B) (0, 0) = 0$
- $r_(2, 0) = (dif^2 r)/(dif A^2) (0, 0) = 0$
- $r_(0, 2) = (dif^2 r)/(dif B^2) (0, 0) = 0$

They're all $0$, except for $r_(1, 1)$, which is $r(A, B) = k A B$.

So the system is:

$
  cases(
    dot(A) = -k A B,
    dot(B) = -k A B,
    dot(C) = k A B,
  )
$

Now, what if we had $2A -> C$? We would gt $dot(A) = -2 k A^2$ and $dot(C) = k A^2$. The $2$ we see is a _stoichiometric coefficient_.

The general version for $sum alpha_n A_n -> sum beta_n B_n$, we get that $r = k product A_n^(alpha_n)$, $dot(x) = plus.minus nu r$ for a stoichiometrix coefficient $nu$, and for a system of reactions, the rates add.

So, for $alpha A + beta B -> gamma C + delta D$, we would have $r = k A^alpha B^beta$ and:
- $dot(A) = -alpha r = -alpha k A^alpha B^beta$
- $dot(B) = -beta r = -beta k A^alpha B^beta$
- $dot(C) = gamma r = gamma k A^alpha B^beta$
- $dot(D) = delta r = delta k A^alpha B^beta$

#example[
  $ 2A harpoons.ltrb^(k_1)_(k_2) A $

  We get:

  $r_1 = k_1 A^2$
  $r_2 = k_2 A$

  $ dot(A) = -2r_1 + r_1 - r_2 + 2r_2 = r_2 - r_1 = A(k_2 - k_1 A) $

  We can rewrite it as:

  $ dot(A) = k_2 A (1 - A/M) $

  for $M = k_2/k_1$. This is the logistic equation!
]

#example[
  Let's try it with the reaction

  $
    A + B ->^(k_1) 2B \
    A + B ->^(k_2) 2A
  $

  The constants are $r_1 = k_1 A B$ and $r_2 = k_2 A B$.

  The equations are:

  $
    dot(A) = -r_1 -r_2 + 2r_2 = r_2 - r_1 = (k_2 - k_1) A B \
    dot(B) = -r_1 + 2r_1 - r_2 = r_1 - r_2 = -(k_2 - k_1) A B
  $

  We know that $A + B$ is constant so $B = N - A$ for a number of particles $N$ then $(k_2 - k_1) A (N - A)$. This way we get that

  $
    dot(A) & = (k_2 - k-1) A (N - A) \
           & = k A(1-A/N)
  $

  for $k = (k_2 - k_1) N$. It's the logistic equation again!
]

Let's talk about conservation laws.

#todo[I guess we didn't???]

$ alpha A + beta B ->^k gamma C + delta D $

#todo[Somehow we got to this]

$
  dot(A) = -alpha k A^alpha (beta/alpha A + beta_0 - beta/alpha A_0)^beta
$

#example[Michaelis-Menten Kinetics][
  This models the reaction with enzymes. The image we have is:

  $ S + E harpoons.rtlb^(k_1)_(k_(-1)) C ->^(k_2) P + E $

  We get equations:

  $
    dot(S) & = -k_1 S E + k_(-1) C \
    dot(E) & = -k_1 S E + k_(-1) C + k_2 C \
    dot(C) & = k_1 S E - k_(-1) C - k_2 C \
    dot(P) & = k_2 C
  $

  We can get two conservations laws by adding up the equations. Namely:

  $
             dot(E) + dot(C) & = 0 \
    dot(S) + dot(C) + dot(P) & = 0
  $

  And we need some initial conditions, so let's assume we start with only sustrate and enzymes, so $S(0) = S_0$, $E(0) = E_0$, $P(0) = 0$ and $C(0) = 0$. We can write the previous conversation laws with these quantities as:

  $
    E + C = E_0 => E = E_0 - C \
    S + C + P = S_0 => P = S_0 - S - C
  $

  Substituting these into the first equations, we get:

  $ dot(S) = -k_1 S (E_0 - C) + k_(-1) C = k_(-1) C - (k_1 E_0) S + k_1 S C $
  $
    dot(C) = k_1 S(E_0 - C) - (k_(-1) + k_2) C = k_1 E_0 S - (k_(-1) + k_2) C - k_1 S C
  $

  There is a twist to this story. One of these reactions (the second one) is being produced much faster than the other, so we can assume that it is always at equilibrium (where $dot(C) = 0$) before the first one. At that point we can express $C$ as a function of $S$ as $C = (k_1 E_0) S div (k_(-1) + k_2 - k_1 S)$.

  Basically, we can see this equation as some substrate becoming a product. But this is not mediated by the law of mass acction because there is a non-linear dependence on $S$ because of the fact that enzymes are limited.
]

This all comes back to the law of mass action for Lotka-Volterra. The reactions with the rabbits and the lynxes would be:

$
  R ->^a 2R \
  R + L ->^b 2L \
  L ->^c emptyset
$

The original equations are just the law of mass action, with $r_1 = a R$, $r_2 = b R$, $r_3 = c L$:

$
  dot(R) & = overparen(a R, r_1) - overparen(b R L, r_2) \
  dot(L) & = underparen(b R L, r_2) - underparen(c L, -r_3)
$

== Example: Harmonic oscillator revisited

We have a harmonic oscillator with equations:

$
  cases(
    dot(x) = v,
    dot(v) = -omega^2 x
  )
$

We can write this as a matrix $dot(x) = A x$ where:

$ x = mat(x; v) quad A = mat(1, 0; -omega^2, 0) $

This results in conentric ellipses going clockwise, with $0$ as a stable point. In this case it is called a _centre_. It is also in this case _marginally stable_.

== Example: Simple example?

When we have a system like $dot(x) = A x$ where $x = mat(x_1; x_2)$ and $A = mat(a, b; c, d)$, if the solution is of the form $x(t) = e^(lambda t) u$ for a constant vector $u$, then the solution to the equation is:

$
  dot(x) = A x \
  lambda e^(lambda t) u = A e^(lambda t) u
  A u = lambda u
$

So the solutions are the eigenvectors and eigenvalues.

So let's solve a general case.

$
  det(A - lambda I) = det mat(a - lambda, b; c, d - lambda) = lambda^2 - underbrace((a + d), tau = h.bar A) lambda + underbrace(a d - b c, Delta = det A)
$

We can decompose it as

$ V^(-1) A V = C $

And we have that $h.bar A = h.bar C$ and $det A = det C$

So, the characteristic polynomial is:

$ p(lambda) = lambda^2 - tau lambda + Delta $

The solutions $p(lambda) = 0$ are when $lambda_plus.minus = (tau plus.minus sqrt(tau^2 - 4 Delta))/2$

Depending on the discriminant we can have different results.

+ $tau^2 > 4 Delta$
  Here we have two eigenvalues: $lambda_1, lambda_2 in R$ where $lambda_1 != lambda_2$.

  We have that $A v_j = lambda_j v_j$. Let's take $V = mat(v_1, v_2)$. If we take $A V$ we get
  $
    A V = (A v_1, A v_2) = (lambda_1 v_1, lambda_2 v_2) = V D
  $

  for $D = mat(lambda_1, 0; 0, lambda_2)$.

  So then we can write $x(t) = z_1(t) v_1 + z_2(t) v_2$ as a change of basis.

  #todo[This goes somewhere $z_j (t) = c_j e^(lambda_j t)$]

  And we can do more stuff:

  $ V dot(Z) = A V Z = V D Z -> dot(Z) $

  #example[
    $
      cases(
        dot(x) = 11x + 2y,
        dot(y) = 12x + 9y
      )
    $

    $ A = mat(11, 2; 12, 9) $

    $ tau = 20, Delta = 99 - 24 = 75 $

    $ tau^2 - Delta = 400 - 300 = 100 $

    $ lambda = (20 plus.minus 10)/5 = cases(5, 15) $

    With the eigenvalues, we can compute the eigenvectors:

    $ (A - 15I bar 0) = mat(-4, 2; 12, -6 | 0) => v_1 = mat(1; 2) $
    $ (A - 5I bar 0) = mat(6, 2; 12, 4 | 0) => v_1 = mat(1; -3) $
  ]

+ $tau^2 = 4 Delta$. In this case we get one eigenvector. Imagine we get an eigenvalue $lambda_0$ with eigenvector $v_0$. Let's take another vector $v_2$ not parallel to $v_0$. Then, we can take
  $
    v_1 = (A - lambda_0 I) v_2 = alpha v_2 + beta v_0 \
    (A - lambda_0 I) v_1 = alpha (A - lambda_0 I) v_2 + beta cancel((A - lambda_0 I) v_0) \
    A v_1 - lambda_0 v_1 = alpha v_1 \
    A v_1 = (lambda_0 + alpha) v_1 \
  $

  Eventually we reach:

  $ A v_2 = v_1 + lambda_0 v_2 $

  And

  $
    A V = mat(A v_1, A v_2) = mat(lambda_0 v_1, v_1 + lambda_0 v_2) = V underbrace(mat(lambda_0, 1; 0, lambda_0), J \ "(Jordan's" \ "canonical" \ "form)")
  $

  The main equation we need is the fact that

  $ A V = V J $

  We can now do a change of variables again $x(t) = z_1(t) v_1 + z_2(t) v_2 = V z$, so that:

  $
    dot(x) = V dot(z) = A x = A V z = V J z \ => V dot(z) = V J z \ => dot(z) = J z
  $

  This is not a totally decoupled system, but it's the closest we can get. The system explicitly is:

  $
    cases(
      dot(z_1) = lambda_0 z_1 + z_2,
      dot(z_2) = lambda_0 z_2
    ) => cases(
      z_2 = c_2 e^(lambda_0 t),
      z_1 = (c_1 + c_2 t) e^(lambda_0 t)
    )
  $

  which has easy solutions, as shown.
+ $tau^2 - 4 Delta < 0$
  $lambda_(plus.minus) = tau/2 plus.minus i omega$ where $omega = sqrt(4Delta - tau^2)$

  $lambda_- = overline(lambda_+)$


  $ x(t) = xi(t) omega + overline(xi)(t) overline(omega) $

  We eventually reach:

  $ dot(x) = A x quad V dot(xi) = A V xi = V D xi $ where $ D = mat(tau/2 + i omega, 0; 0, tau/2 - i omega) $

  $
    dot(xi) = D xi = cases(
      dot(xi) = (tau/2 + i omega) xi,
      dot(overline(xi)) = (tau/2 - i omega) overline(xi)
    )
  $

  So somehow $xi(t) = xi_0 e^(tau/2 t) e^(i omega t)$

  And even more eventually:

  $ x(t) = a/2 2 e^(tau/2 t) Re(e^(i(omega t - phi) (u + i v))) $

  $ x(t) = a e^(tau/2 t) (cos(omega t - phi) u - sin(omega t - phi) v) $

  This is a spiral, attractive if $tau < 0$ and repelling if $tau > 0$. The real part tells you that something is being attractive or repelled, the imaginary part tells you how it's rotating. In fact, this applies even when the imaginary part is $0$, because then it's just not rotating. The semantics are conserved.

  However, the phase-space diagram has no clear geometric relationship with the eigenvectors themselves.

  This is called a _focus_, either unstable ($tau>0$) or stable ($tau<0$).


#theorem[
  For a system like

  $
    cases(
      dot(x) = f(x, y),
      dot(y) = g(x, y)
    )
  $

  Where $dot(x) = F(x)$ and $D F(x)$ is continuous in an open set $U$, then there is a set for $(*)$ for every initial condition $x(t_0) = x_0 in U$ and the solution is unique.
]

== A few definitions

#definition[Nullclines][
  Nullclines are the manifolds where the derivatives are $0$. They divide the plane in positive and negative $dot(x)$ and $dot(y)$, for instance. In nullclines, the trajectories cross in an axis direction (say either vertical or horizontally).
]

#definition[Homoclines and heteroclines][
  Homoclines are trajectories that start and end at the same fixed point. Heteroclines are trajectories that start at a fixed point and end at a different fixed point. These are important because they also divide the plane in important regions.
]

#definition[Stability][
  We say a point $x^*$ is a stable fixed point if for every $R>0$ there exists $0 < r <= R$ such that if $norm(x(t_0) - x^*) <= r$ then $norm(x(t) - x^*) <= R$ for all $t > t_0$.
]

#definition[Asymptotical stability][
  A fixed point $x^*$ is called _asymptotically stable_ or an _attractor_ if for some $R > 0$ then if at some point $norm(x(t_0) - x^*) <= R$ then $lim_(t->oo) x(t) = x^*$
]

The difference between the two definitions above is that the first only guarantees that if you start sufficiently close to the fixed point you stay in the fixed point, while the second says that if you start close to the fixed point you start getting _closer_ to the fixed point.

Let's examine fixed points. For $dot(x) = F(x)$, a point $x^*$ being fixed means that

$
  F(x) = cancel(F(x^*)) + overbrace(D F (x^*), A) (x - x^*) + o(norm(x - x^*)) \
  dot(x) = A (x - x^*) + ...
$

If $A$ is nonsingular then $x^*$ is a fixed point. The eigenvalues will tell you whether the points are attracting, repelling or saddle, and the eigenvectors in what directions.

However, we still have nonlinear terms! To see what happens we need to look at the $tau^2$ vs $4Delta$ diagram. If it is well on the stable diagram, it stays stable and the local analysis above works fine. The problem is if the stable point is a stable center (at the y-axis), because any tiny perturbation can make it either stable or unstable.

#example[
  Let's take

  $
    cases(
      dot(x) =-x + x^3,
      dot(y) = -2y
    )
  $

  The nullclines are at $x=0$, $x = plus.minus 1$ and $y = 0$.

  This results in three fixed points. We can analyze $(0, 0)$ by linearizing, obtaining $dot(x) = -x, space dot(y) = -2y$ so $A = mat(-1, 0; 0, -2)$.

  The Jacobian is $D F = mat(-1 + 3x^2, 0; 0, -2)$ so $D F(plus.minus 1, 0) = mat(2, 0; 0, -2)$
]

#example[Rabbits vs sheep][
  Let $x$ be rabbits, $y$ be sheep, and

  $
    cases(
      dot(x) = x(3 - x - 2y),
      dot(y) = y(2 - x - y),
    )
  $

  This is a typical scheme of competition.

  The nullclines are at $x = 0$, $x + 2y = 3$, $y = 0$ and $x + y = 2$. To get all fixed points we need to combine these equations. In the end we get $(0, 0)$, $(0, 2)$, $(3, 0)$, $(1, 1)$.

  Taking the jacobian we have:

  $ D F = mat(3 - 2x - 2y, 2x; -y, 2 - x - 2y) $

  and we get

  - $D F (0, 0) = mat(3, 0; 0, 2)$ $->$ unstable
  - $D F(0, 2) = mat(-1, 0; -2, -2)$ $->$ stable
  - $D F(3, 0) = mat(-3, -6; 0, -1)$ $->$ stable
  - $D F(1, 1) = mat(-1, 2; -1, -1)$. Trace is $tau=2$, determinant $Delta = 1 - 2 = -1$ so this is a saddle point.

  If we plot this we get that rabbits and sheep can't coexist (this is called competitive exclusion).
]

== Example: Transformation of a centre

$
  cases(
    dot(x) = -y + a x(x^2 + y^2),
    dot(y) = x + a y(x^2 + y^2)
  )
$

We should use colar coordinates here, so we have $r$ and $theta$ where $x = r cos(theta)$, $y = r sin(theta)$ and $x^2 + y^2 = r^2$. Also $tan theta = y/x$.

We need to get $dot(r)$ somehow, and we do it by $2 r dot(r) = 2x dot(x) + 2y dot(y)$ so $ dot(r) = x/r dot(x) + y/r dot(y) $

We also have that $(1 + tan^2 theta) dot(theta) = (theta dot(x) ) / x^2$

$ dot(theta) = (x dot(y) - y dot(x)) / r^2 $

Replacing these into the equations, we get:

$
  dot(r) & = cos theta (cancel(-r sin theta) + a r^3 cos theta ) + sin theta (cancel(r cos theta) + a r^3 sin theta )
  \ & = a r^3 (cos^2 theta + sin^2 theta)
  \ & = a r^3
$


$
  dot(theta) & = -1/r^2 r sin(theta)(-r sin theta + a r^3 cos theta) + 1/r^2 r cos theta (r cos theta + a r^3 sin theta)
  \ & = sin^2 theta + cos^2 theta
  \ & = 1
$

So we only have a single nullcline at $r=0$ so the only fixed point is at $(0, 0)$.

We can linearize the system and we get just $dot(x) = x$ and $dot(y) = y$ so the Jacobian is $D F = mat(0, -1; -1, 0)$, trace is $0$, determinant is $1$ and thus we cocnlude that the fixed point is a centre. Uh oh, we cannot just linearize then, we have to see if we're in a stable or unstable point!

We can see it by eye. If $a < 0$ then it is an attracting spiral, if $a > 0$ it's a repelling spiral and only if $a = 0$ it is actually a center.

== Example: Conservative systems

If we have something like $m dot.double(x) = F(x) = -V'(X)$ then $m dot.double(x) + V'(x) = 0$ And then we get $m dot(x) dot.double(x) + V'(x) dot(x) = 0$ and finally

$
  m dif / (dif t) (1/2 dot(x)^2) + dif/(dif t) V(x) = 0 \
  dif/(dif t) (1/2 m dot(x)^2 + V(x)) = 0
$

We essentially get a nontrivial equation that is conserved along the dynamics. Namely,
$ E(x, dot(x)) = 1/2 m dot(x)^2 + V(x) $

And we name it $E$ because in this context, it is the energy, and energy is conserved.
Let's take the case $V(x) = -1/2 x^2 + 1/4 x^4$. The equations would be $ cases(
  dot(x) = y,
  dot(y) = x - x^3,
) $

So $E = 1/2 y^2 - 1/2 x^2 + 1/4 x^4$ is conserved.

These systems cannot have attractors unless the constant is completely constant. Nodes and focuses are excluded. All they can have are centers and saddles. Centers are the minima of the conserved quantities, and the saddles are maxima.

In this particular case, these trajectories are homoclinic.

#definition[Conservative system][
  We say that a system $dot(x) = F(x)$ is _conservative_ if there exists a non-trivial quantity $E(x)$ such that $dif/(dif t) E(x) = 0$.
]

== Index theory

We want to analyze the _index_ of a curve.

#definition[Index number of a curve][
  THe _index number_ $I_c$ of a curve is defined as

  $
    I_c = [Delta phi]_c/(2pi)
  $

  Essentially, we see how many turns we take if we evaluate the vector field along a curve.
]

#theorem[Index number properties][
  + If the simple closed curve $c_1$ can be continuously deformed into $c_2$ without ever crossing a fixed point, then $I_(c_1) = I_(c_2)$.
  + If there are no fixed points in the interior of $c$ then $I_c = 0$ (this follows from the previous, as you can essentially reduce the curve to a single point).
  + Replacing $F(x)$ by $-F(x)$ doesn't change $I_c$.
  + If $c$ is a trajectory of the dynamical system $dot(x) = F(x)$ then $I_c = 1$.
]

These all are pretty sensible.

#definition[Index of a point][
  The _index_ of a point $x$ is the inde of any closed curve surrounding $x$ and no other fixed point.
]

So, the index of a non-fixed point is always $0$. The index of a fixed point $x^*$ is $1$ unless it is a saddle point, in which case it is $-1$.

#theorem[
  If a simple closed curve $c$ surrounds $n$ isolated fixed points $x^*_1, ..., x^*_n$, then the index of this curve is the sum of the indices of these points. That is:

  $ I_c = sum_(j=1)^n I_j $
]

This is a pretty important theorem. It's easy to see that this is the case because you can take the curve and deform it to hug them with trajectories that are arbitrarily close to the points or to the "highways" between points.

== Example: Van der Pol's oscillator

The equation is:

$ dot.double(x) + mu(x^2 - 1) dot(x) + x = 0 $

This is like a damped oscillator when $x < 1$ but it actually amplifies oscillation when $x > 1$. We are going to study the cases where $mu$ is large and $mu$ is small.

== Gradient systems

We are going to take a look at _gradient systems_ which have no closed orbit. We have that

$ dot(x) = F(x) -gradient V(x) $

We have that $dot(x) = f(x, y) = -(partial V)/(partial x)$ and $dot(y) = g(x, y) = -(partial V)/(partial y)$. We can check if these form a gradient if $(partial f)/(partial y) = (partial g)/(partial x)$.

If there is a closed orbit, we would have that $x(0) = x(T)$ for some period $T > 0$.

$
  V(x(0)) = V(x(T))
$

so

$
  V(x(0)) - V(x(T)) = 0 \
  integral_0^T dif/(dif t) V(x) dif t \
  integral_0^T gradient V dot(x) dif t \
  =^(dot(x) = -gradient V) - integral_0^T norm(gradient V)^2 dif t \
  => norm(gradient V) = 0 \
  => gradient V = 0
$

which implies that a gradient system has a closed orbit only if the potential is constant, which is a trivial system.

So, checking if a system is gradient is a way to reach conclusion about closed orbits.
== Lyapunov functions

Here's an example:

$
  cases(
    dot(x) = y,
    dot(y) = -x - y^3
  )
$

This system is clearly not gradient.

If we differentiate $dot(x)$ we can rewrite the system as $dot.double(x) = -x - dot(x)^3$ or equivalently:

$
  dot.double(x) + x = -dot(x)^3
$

Let's multiply by $dot(x)$:

$
  underbrace(dot(x) dot.double(x) + x dot(x)) & = -dot(x)^4 \
         dif/(dif t) (1/2 dot(x)^2 + 1/2 x^2) & = -dot(x)^4
$

We can write this as a function $ E(x, dot(x)) = 1/2(x^2 + dot(x)^2) $

and as such

$
  E(x(T), dot(x)(T)) - E(x(0), dot(x)(0)) = -integral_0^T dot(x)^4 dif t <= 0
$

#definition[Lyapunov function][
  Let $dot(x) = F(x)$ and $x^*$ be a stable point such that $F(x^*) = 0$. We say that $E(x)$ is a _Lyapunov function_ for this system if:
  + $E(x)$ has continuous derivatives in an open set containing $x^*$.
  + $E(x) > 0$ for all $x != x^*$ and $E(x^*) = 0$ (i.e., the fixed point is a local minimum of this function).
  + $gradient E(x) dot F(x) < 0$ for all $x != x^*$.
]

Let's check that $E$ is a Lyapunov function. We have:

+ $E(x, y) = 1/2 (x^2 + y^2) >= 0$
+ $E(x^*) = 0$
+ $gradient E dot F = x y + y (-x - y^3) = -y^4 <= 0$

So it is!

If we have a Lyapunov function we can show that there are no stable orbits by the same argument as for the gradient systems. To be clear:

$
  E(x(T)) - E(x(0)) = integral_0^T dif/(dif t) E(x, y) dif t = integral_0^T gradient E dot F dif t <= 0
$

and this can only happen if $gradient E dot F = 0$. In the case the inequality is strict there is just no periodic orbit at all, and since $-y^4 = 0$ only when $y = 0$ and the stable point is at $(x, y) = (0, 0)$, then in this case there are no periodic orbits at all.

=== Example: Damped harmonic oscillator

The equation is:

$ dot.double(x) + mu dot(x) + V'(x) = 0 $

If $mu = 0$, then $dot.double(x) + V'(x) = 0$. We can define $E = 1/2 dot(x)^2 + V(x)$ so $(dif E)/(dif t) = 0$. We can rewrite this as:

$
  cases(
    dot(x) = y,
    dot(y) = -mu y - V'(x)
  )
$

And then we rewrite $E$ as $ E(x, y) = 1/2 y^2 + V(x). $

We want to show that $E$ is a Lyapunov function, so:
+ $E$ is differentiable
+ $E(x^*, 0) = V(x^*) = 0$ and $E(x, y) > 0$ in an environment of $(x^*, 0)$.
+ $gradient E = (V'(x), y) => gradient E dot F = V'(x) y + y(-mu y - V'(x)) = -mu y^2 < 0$ when $y != 0$.

$E$ is indeed a Lyapunov function, which shows that these systems cannot have closed orbits. This is interesting because without the friction we have all closed orbits, but with friction they become attractors.

=== Example:

Applying Lyapunov functions can be tricky.

$
  cases(
    dot(x) - y -x^3,
    dot(y) = x - y^3,
  )
$

$x = y^3$, $y = -x^3$ so $(0, 0)$ is the onnly fixed point.

A Lyapunov function is $E(x, y) = x^2 + y^2$. The first two conditions are easily met. As for the last one,

$
  gradient E dot F & = 2x (-y - x^3) + 2y (x - y^3) \
                   & = -2x^4 - 2y^4
$

which is clearly positive if $x != and y != 0$, so $E$ is a Lyapunov function, so the system has no closed orbits.

If you have a general system, finding Lyapunov functions is more of an art than a science. What you can try sometimes do is take something of the form:

$
  E(x, y) & = a(x - x^*)^2 + b(x - x^*)(y - y^*) + c(y - y^*) \
          & = (x - x^*, y - y^*) mat(a, b/2; b/2, c) mat(x - x^*; y - y^*)
$

You want $a > 0$ and the matrix to be positive definite for the gradient to go away around the fixed points. Since $a > 0$, positive definiteness implies that the determinant must be positive, which is given by $a c - b^2/4$ or, as it is more common to write it,

$ b^2 - 4a c < 0. $

This can rule out closed orbits _in a region_ but not necesarilly everywhere.

== Poincaré-Bendixson Theorem

Now, how could we show that a system _does_ have stable orbits?

#theorem[Poincaré-Bendixson][
  If
  + $R$ is a closed bounded subset of a plane
  + $dot(x) = f(x)$ is a continuously differentiable vector field on an open set containing $R$
  + $R$ does not contain any fixed point and
  + There exists a trajectory $C$ that is "confined" in $R$

  then, either $C$ is a closed orbit or there is a closed orbit in $R$.
]

Intuitively, this makes sense because if there are no fixed points in a region and trajectories cannot leave

To apply this theorem we need to find a trapping region without a fixed point.

If you have a repeller, it is easy to find a trapping region since you can take a big region containing the repeller and cutting out a small region around the repeller, which will create a trapping region.

=== Example: Limit cycles

Take system:

$
  cases(
    (dif r)/(dif t) = r(1 - r^2) + mu r cos theta,
    (dif phi)/(dif t) = 1
  )
$

We have that $-1 <= cos theta <= 1$ so the first equation is bounded by:

$ r(1 - r^2 - mu) <= r(1 - r^2) + mu r cos theta <= r(1 - r^2 mu) $

So then,

$
  r_"min" (1 - r^2_"min" - mu) > 0 & quad => quad & r_"min" < sqrt(1 - mu) \
  r_"max" (1 - r^2_"max" + mu) < 0 & quad => quad & r_"max" > sqrt(1 + mu) \
$

We can take $r_"min" = 0.99 sqrt(1 - mu)$ and $r_"max" = 1.01 sqrt(1 + mu)$ and this way we find a trapping region.

=== Example: Glycolysis (Sel'kov model)

The system is

$
  cases(
    dot(x) = -x + a y + x^2 y,
    dot(y) = b - a y - x^2 y,
  )
$

where $x$ represents ADP (the pre/postcursor to ATP) and $y$ is F6P (basically fructose).

We want to show that this has a periodic orbit.

The first thing we do is plot the nulclines.

#{
  let a = 1.2
  let b = 0.8
  figure(
    lq.diagram(
      lq.plot(lq.linspace(0, 10), x => (x * x) / (a + x * x)),
      lq.plot(lq.linspace(0, 10), x => b / (a + x * x)),
    ),
  )
}

Then we find the arrows in each of the regions.

We have a horizontal line at the top that makes one bound of the region and one on the right. This is because it is formed at the points where $dot(x) = 0$ and $dot(y) = 0$ #todo[or maybe I got it wrong]. Then, we need to calculate the slope of these two:

$
  dot(y)/dot(x) = (dif y)/(dif x) = (b - y(a+x^2))/(y (a+x^2) - x) = -(y(a + x^2) - b)/(y(a + x^2) - x)
$

This is $<= -1$ if $x >= b$. So then, the trapping region is closed by a slope of $-1$ because all trajectories have, at that point, slope $<= 1$. So they go all inside.

We now have a trapping region but we still have a stable point inside. The fixed point is at $x^* = (b, b/(a + b^2))$. We need to determine its stability, and we do this with the Jacobian:

$
  & D F = mat(-1 + 2x y, a + x^2; -2x y, -(a+x^2)) \
  => & D F(x^*) = mat(-1 + 2 b^2/(a+b^2), a + b^2; -2 b^2 / (a + b^2), -(a + b^2))
$

We now calculate $Delta = det(D F(x^*)) = a-b^2 + 2b^2 = a + b^2 > 0$. The determinant is positive so the stability of the point depends on the trace, which is:

$
  tau = (b^2 - a)/(a + b^2) - (a + b^2) = (b^2 - a - (a + b^2)^2)/(a + b^2) > 0
$

And thus the point is unstable iff $b^2 - a - (a + b^2)^2 > 0$. Let's work through this a bit more:

$
                b^2 - a - (a + b^2) & > 0 \
              (a + b^2)^2 + a - b^2 & < 0 \
      a^2 + b^4 + 2 a b^2 + a - b^2 & < 0 \
      a^2 + b^4 + 2 a b^2 + a - b^2 & < 0 \
  a^2 + a(2b^2 + 1) + b^2 (b^2 - 1) & < 0 \
$

This is a parabola, and for this to be negative it needs to be between the two roots:

$
  2 a_plus.minus & = -2b^2 - 1 plus.minus sqrt((2b^2 + 1)^2 - 4b^4 + 4b^2) \
                 & = -2b^2 - 1 plus.minus sqrt(4b^4 + 4b^2 + 1 - 4b^4 + 4b^2) \
                 & = -2b^2 - 1 plus.minus sqrt(8b^2 + 1) \
$

the condition we want is $0 < a < a_+$ so the condition is:

$
  2a < sqrt(1 + 8^2) - 1 - 2b^2
$


== Example: Relaxation oscillations (Van der Pol oscillator, $mu >> 1$)

Revisited. Equations are:

$ dot.double(x) + mu(x^2 - 1) dot(x) + x = 0 $

where we can write $dot(x)$ as $dot(x) = mu (y - G(X))$ with $G(x) = x^3/3 - x$. We can define a new variable $w$ where $dot(w) = -x$, and $dot(w) + x = 0$ and we get

$
  cases(
    dot(w) = -x,
    dot(x) = w - mu G(x)
  )
$

and with $w = mu y$ we end up with

$
  cases(
    dot(x) = mu (y - G(x)),
    dot(y) = -1/mu x,
  )
$

The idea here is that we are going to assume that $mu >> 1$. This produces relaxation oscillations, where it cycles and in each cycle slowly moves at first and then jumps to the other end.

If $mu$ is very large, $1/mu$ is small so $y$ increases slowly and $x$ increases very fast. It approaches a cycle around the nullcline.

We can calculate the period:

$
  T = 2 integral_(t_D)^(t_A) dif t + 2 integral_(t_A)^(t_B) dif t \
  T = underbrace(2 / mu, approx 0) integral_(x_D)^(x_A) (dif x)/(y - G(x)) + 2 integral_(y_A)^(y_B) dif t \
  T = 2 integral_(y_A)^(y_B) dif t \
$

This second integral is close to the nullcline, that is, when $y = G(x) + 1/mu^2 H(x)$ and $dot(y) = (G'(x) + 1/mu^2 H'(x)) = -x/m$ where we get that $dot(x) = -x/mu G'(x) (1 + O(1/mu))$

And eventually we get

$
  integral_(t_A)^(t_B) dif t = mu integral_(x_A)^(x_B) (-(G'(x))/x) dif x + O(1/mu)
$

We get $x_B$ at the extrema of $G$, which are at $plus.minus 1$. Let's take $x_B = 1$. For $x_A$ we can approximate it as the coordinate where a straight line from $D$ intersects $G(x)$. So, $G(-1) = 2/3$ and then $G(x) = 2/3$ happens when $x = 2$.

The integral evaluates to:

$
  integral_(t_A)^(t_B) dif t = O(1/mu) + mu (3 - 2 ln 2)
$

So, if $mu$ is very large, $T = mu (3 - ln 4)$.

== Example: Duffing equation

#faint[Page 239 of Strogatz.]

The equation is

$ dot.double(x) + x + epsilon x^3 = 0. $

We can write this as $x = x_0 + epsilon x_1 + epsilon^2 x_2 + ...$. However, this would be very wrong! This is because this problem has two time scales: the period of rotation and the time it takes to reach the limit.

Let's take two times, $tau = t$ and $T = epsilon t$. The assumption we make here is that $x(t)$ can be written as $x(tau, T)$.

So, we can write

$ x = x_0(tau, T) + epsilon x_1 (tau, T) + ... $

Taking derivatives of $x$ is now a bit more complicated. Namely:

$ dot(x) = x_tau + x_T epsilon $
$ dot.double(x) = x_(tau tau) + 2 epsilon x_(tau T) + epsilon^2 x_(T T) $

This is what makes this a bit more tricky. #faint[Here is where the fun starts.]

#faint[Page 243 from Strogatz.]

We are going to neglect everything that contains $epsilon^2$ or higher.

$
  x = underbrace((x_(0 tau tau) + 2 epsilon x_(0 tau T) + ...) + epsilon (x_(1 tau tau) + ...), x_0) \
  + underbrace(x_0 + epsilon x_1 + ..., x_1) \
  + epsilon(x_0^2 + 1) x_(0 tau) + ... = 0
$

we can collect same order terms and set them to $0$.
- For order 1, $x_(0 tau tau) + x_0 = 0$, which is a harmonic oscillator with general solution $r(T) cos(tau + phi(T))$
- For order $epsilon$
  $ x_(1 tau tau) + x_1 = -2 x_(0 tau T) -(x_0^2 - 1) x_(0 tau) $

  One side is
  $
    x_(0 tau T) & = partial/(partial T) (-r(T) sin(tau + phi(T))) \
                & = -r' sin(tau + phi) - r phi' cos(tau + phi) \
  $
  and the other side
  $
    (x_0^2 - 1) x_(0 tau) = (r^2 cos^2(tau + phi) - 1)(-r sin(tau + phi))
  $

  Eventually we get


== Example: Griffith's model

#faint[Page 269 from Strogatz.]

== Example: Pitchfork bifurcation: Overdamped anharmonic oscillator

Eventually we get:

$ E = k/2 ((l_c - l_0)/l_c x^2 + l_0/(4l_c^3) x^4 + ...) $

Here we introduce $xi = (l_c - l_0) / l_c$. If $xi < 0$, only fixed point is $x = 0$; if $xi > 0$ then two fixed points $x^*_plus.minus$.

Using simplifying assumption $l_0 approx l_c$, then we have that

$ E approx k/2 (xi x^2 + x^4/(4l_0^2)) $

This is the equation we will use.

We can differentiate to get maxima and minima, which are $x = plus.minus l_0 sqrt(2xi)$.

The equation of motion is

$
  m dot.double(x) + mu dot(x) + (partial E)/(partial x) & = 0 \
    m dot.double(x) + mu dot(x) - k xi x + k/(2l_0) x^3 & = 0 \
  dot.double(x) + mu/m dot(x) - k xi x + k/(2l_0 m) x^3 & = 0 \
$

We can change the scale of $x$. Let's introduce the natural scale of length $x_c$ and of time $t_c$ and take:

$
  cases(
    x = x_c u,
    t = t_c tau,
    xi = q epsilon,
  )
$

Put it into the equation...

$
  x_c/(t_c^2) u'' + (mu x_c)/(m t_c) - (k q x_c)/(m) epsilon u + (k x_c^3)/(2l_0^2 m) u^3 = 0 \
  u'' + (mu t_c)/m u' - (k f t_c^2)/m epsilon u + (k x_c^2 t_c^2)/(2l_0^2 m) u^3 = 0
$

Here we get

$
  t_c = m / mu \
  x_c = l_0 mu sqrt(2/(k m)) \
  q = mu^2/(k m)
$

And this gets converted to...

$ u'' + u' - epsilon u + u^3 = 0 $

You can actually do this for any potential that is symmetric like this, because we truncated the terms. This equation, then, is kind of universal.

For $epsilon < 0$, fixed point is $u^* = 0$ and if $epsilon > 0$ we have $u^* = 0$, $u^*_plus.minus sqrt(epsilon/3)$

Now, let's study the dynamics.

For $epsilon < 0$, we have that $u'' + u' - epsilon u = 0$ (we get rid of the $u^3$ because $u$ is small?). The characteristic function is $lambda^2 + lambda - epsilon = P(lambda)$, the roots are $lambda_plus.minus = (-1 plus.minus sqrt(1 + 4epsilon))/2 approx epsilon$ as $epsilon$ is small. The solutions are then:

$
  u = c_1 e^(-tau) + c_2 e^(epsilon tau)
$

Here we get the two scales of time. The $e^(-tau)$ disappears much faster than $e^(epsilon tau)$. As $tau$ increases, the term that defines the behaviour is the $e^(epsilon tau)$ because $e^(-tau) -> 0$ fast.

The overall behavior is that you go fast down the y axis and then slowly approach the origin. Whenever you have a fast and a slow mode, the same thing happens. The point is that the points quickly go to a one dimensional manifold so a pitchfork bifurcation (which is what we have here) is essentially a one dimensional phenomenon.

Ok, so let's try to introduce the slow scale. Let $T = epsilon tau$. We are going to assume that $u$ is given by an expansion. So,

$ u = epsilon^1/2 A_0(T) + epsilon A_1(T) + epsilon^(3/2) A_2 (T) + ... $

Eventually, we get

$ A'_0 = A_0 - A_0^3 $

Which is the normal form of a pitchfork bifurcation. So, once the fast mode has ended, we get a pitchfork bifurcation in the one dimension it has covnerged to.

Since the equation of $u'' + u' - epsilon u + u^3 = 0$ is universal, then $A'_0 = A_0 - A_0^3$ is also universal! The second derivative $u'' = O(epsilon^(5/2))$ is negligible.

== Theory: Hopf bifurcation

The equations of Hopf bifurcations are

$
  cases(
    dot(R) = mu r - r^3,
    dot(theta) = omega + b r^2
  )
$

#faint[Page 274 of Strogatz.]

And, very eventually, you get this:

$
  cases(
    rho' = rho (nabla_r - ell_r/2 rho^2),
    phi' = nabla_i - ell_i/2 rho^2,
  )
$

Where $A = rho e^(i phi)$, and $r$/$i$ subscripts indicate real and imaginary parts.

This equation is universal again, so every pitchfork bifurcation follows these equations.


== One-Dimensional Map

This is how you get the logistic map and the mandelbrot set.

#faint[Page 385 of Strogatz.]

== Fractals

#faint[Page 435 of Strogatz.]

= Markov processes and chains

A Markov chain is a Markov process with discrete state space.

$ P(n m) = P(n, n) P(r, m) $

We have time-homogeneous Markov Chains where the probabilities are the same over time, in the sense that

$ P(n, m) = P(n - m) $

So then

$ P(x_(n+k) = s_i | x_n = s_j) = P_(i j)^((k)) $

And we denote

$ P^((k)) = (P_(i j)^((k))) $

And in fact

$ P(n) = P(1)^n $

We remove the time dependence at some point, so

$ P = P(1) $

and $ P(n) = P^n $


We are going to denote $1 = mat(1; 1; 1; dots.v; 1)$

A funny thing is that $P^n$ converges as $n -> oo$. Let's see why this happens.

We start with the eigenvalues $P v_i = lambda_i v_i$. Let $V = mat(v_1, v_2, v_3)$ and $Lambda = mat(lambda_1, 0, dots.h, 0; 0, lambda_2, dots.h, 0; dots.v, dots.v, dots.down, dots.v; 0, 0, 0, lambda_n)$.

Assume the matrix $P$ is diagonalizable, such that $P V = V Lambda$. Let $U = V^(-1)$ such that $U P = Lambda U$

We get $U = mat(u_1^T; u_2^T; u_3^T; dots.v; u_n^T)$ and $u_i^T P = lambda_i u_i^T$.

Since $U V = I$ we have that $u_i^T v_j = delta_(i j)$. The matrix $P$ can be written now as

$
  P = sum_(i=1)^N lambda_i underbrace(v_i u_i^T, n times n "matrix" \\ "(projector)")
$

The point is that $(v_i u_i^T)^2 = v_i cancel(u_i^T v_i) u_i^T = v_i u_i^T$. The projection stays the same. We can use this in the expression of $P$ and raise it to the $n$ to get:

$ P^n = sum_(i=1)^N lambda_i^n v_i u_i^T $

from which it falls out that

$ P^n = V Lambda^n V^(-1). $

Finally, we can do
$
           1^T P v_i & = lambda_i 1^T v_i \
             1^T v_i & = lambda_i (1^T v_i) \
   "if" lambda_i = 1 & => 1^T v_i = 1 \
  "if" lambda_i != 1 & => 1^T v_i = 0 \
$

== Ergodic and Regular Markov chains

In an Ergodic Markov chain you can reach any state from any other state at most $N$ steps, where $N$ is the size of the state space. That is, $(P^(n_(i j)))_(i j) > 0$.

An Ergodic Markov chain is a regular Markov chain if there exists a $n > 0$ such that $P^n > 0$ (all elements of the matrix are positive).

An Ergodic Markov chain that is not regular is given by two states that go to each other with a probability $1$.

#theorem[Fundamental limit][
  Given a transition matrix $P$ of a regular MC, the limit of $P^n$ as $n$ goese to infinity is a positive matrix. Specifically,

  $
    lim_(n -> oo) P^n = w 1^T = mat(w, w, w, w)
  $

  where $w_i > 0$ and $sum_i w_i = 1$
]

#corollary[
  The only solution of $P v = v$ is a multiple of $w$ and the only solution of $u^T P = u^T$ is a multiple of $1$.
]

#proof[
  $P^2 v = P v = v -> P^2 v = v -> P^n v = v$

  The limit $P^n v = v$ as $n -> oo$ is $v = w (1^T v) prop w$.

  And $u^T P^n = u^T$ and as $n -> oo$ $(u^T w) 1^T = u^T prop 1^T$.
]

#theorem[
  Given an ergodic MC there is a unique probability vector $w$ such that $P w = w$ and $w > 0$. Any vector such that $P v = v$ is a multiple of $w$. Any vector $u$ such that $u^T P = u^T$ is a multiple of $1$.3
]

#theorem[Perron-Frobenius][
  A few.
]

=== Example: Ehrenfest model: sationarity

We have two urns, $N$ total marbles and a $q$ chance of changing one random marble from urn.

The equation $P w = w$ is:

$ w_n = p_(n,n+1) w_(n+1) + p_(n n) w_n + p_(n, n-1) w_(n-1) $

If we think about and replace the probabilities, we get

$
  w_n & = (n+1)/N q w_(n+1) + (1 - q) w_n + (N-n+1)/N q w_(n-1) \
  w_n & = (n+1)/N q w_(n+1) + w_n - q w_n + (N-n+1)/N q w_(n-1) \
  0 & = (n+1)/N q w_(n+1) - q w_n + (N-n+1)/N q w_(n-1) \
  0 & = (n+1)/N w_(n+1) - w_n + (N-n+1)/N w_(n-1) \
  0 & = (n+1)/N w_(n+1) - w_n + (1 - (n-1)/N) w_(n-1) \
  (n+1)/N w_(n+1) - (n-1)/N) w_(n-1) & = w_n - w_(n-1) \
$

Here we using generating functions! Take polynomial

$ Q(z) = sum_(n=0)^N z^n w_n $

We multiply the equation by $sum_n z^n$:

$
  sum_(n=0)^N (n+1)/N w_(n+1)z^n - sum_(n=1)^N (n-1)/N w_(n-1)z^n
  = sum_(n=0)^N w_n z^n - sum_(n=1)^N w_(n-1) z^n
$

Now we shift the indices to get $n$ everywhere:

$
  1/N sum_(n=1)^N (n) w_(n)z^(n+1) - 1/N sum_(n=0)^(N-1) n w_(n)z^(n-1)
  & = sum_(n=0)^N w_n z^n - sum_(n=1)^(N-1) w_n z^(n-1) \
  1/N sum_(n=0)^N (n) w_(n)z^(n+1) - 1/N sum_(n=0)^N n w_(n)z^(n-1) + cancel(1/N z^(N+1) N)
  & = sum_(n=0)^N w_n z^n - sum_(n=1)^N w_n z^(n-1) cancel(z^(N+1)) \
  1/N sum_(n=0)^N (n) w_(n)z^(n+1) - 1/N sum_(n=0)^N n w_(n)z^(n-1) + cancel(1/N z^(N+1) N)
  & = sum_(n=0)^N w_n z^n - sum_(n=1)^N w_n z^(n-1) cancel(z^(N+1)) \
  1/N Q'(Z) - z^2/N Q'(z) & = Q(z) - z Q(z) \
  (1 - z^2) Q'(z) & = N (1-z) Q(z) \
  (1 + z) Q'(z) & = N Q(z) \
  Q(z) = c (1 + z)^N
$

From the construction of $Q$ we can get $Q(1) = sum_(n=0)^N w_n = 1$, so $Q(1) = c (1 + 1)^N => c = 2^(-N)$. Therefore, we find that

$
  Q(Z) = 2^(-N) sum_(n=0)^N z^N mat(N; n)
$

So $ w_n = 1/(2^N) mat(N; n) $

This is just saying that in the stationary state any marble has a $1/2$ chance to be in any urn.


== Absorving Markov chains

An absorving state has $P_(i i) = 1$. That is, once you enter you stay there forever (the state $s_i$ is "absorving"). A non-absorving state is called transient.

An absorving MC contains absorving states but from every transient state you can reach at least one absoring state in a finite number of steps.

Generally you put the transient states at the start and the absorving states at the end. You also denote $t$ transient states and $r$ absorving states.

The limit of $P^n$ for a matrix like this is:

$
  P^n = mat(Q^n, 0; R_n, I)
$


Fundamenta matrix of absorving MC:

$ N = (I - Q)^(-1) = sum_(k=0)^oo Q^k $

This corresponds to the expected number of visits from state $s_j$ to state $s_j$ before absorption, for each $N_(i,j) in N$.

For the drunkyards walk exampe with probabilities $1/2$, we have that:

$ Q = mat(0, 1/2, 0; 1/2, 0, 1/2; 0, 1/2, 0) $
$ (I - Q) = mat(I, -1/2, 0; -1/2, I, -1/2; 0, -1/2, I) $

And the inverse:

$ N = (I-Q)^(-1) = mat(3/2, 1, 1/2; 1, 2, 1; 1/2, 1, 3/2) $

This tells us that the state right next to the absorving states are the least common and the center ones the most common. This tracks with our expectations.

We also have an absorving matrix $B$ where $b_(i j)$ represents the probability to be absorved at $s_i$ if you start at $s_j$.

There is a relation

$
  B = R + B Q => B(I - Q) = R => B = R N
$

In the drunkyard's walk example, we have that

$ R = mat(1/2, 0, 0; 0, 0, 1/2) $

so

$ R N = mat(3/4, 1/2, 1/4; 1/4, 1/2, 3/4) $

== Example: Birth and death process

#show link: set text(fill: blue.lighten(70%))
#link("https://en.wikipedia.org/wiki/Birth%E2%80%93death_process")

Births and deaths, characterized by positive birth rates $lambda_i$ and death rates $mu_i$ (or $b_i$ and $d_i$ in the notes).

$ tau^T = 1^T N $

#todo[What is $tau$?]

$ N = (I - Q)^(-1) $

$n_(i j)$ is the time spent visiting $i$ starting from $j$.

$ tau = mat(tau_1; tau_2; dots.v; tau_t) $

Let $ pi_n = cases(pi_0 = 1, pi_n = (b n)/d_n pi_(n-1)) $

We are solving $tau^T (Q - I) = -1^T$

and we reach

$ pi_n n_(n + 1) - pi_(n-1) n_n = pi_(n-1)/d_n = pi_n / b_n $

This is a telescopic sum, that results in

$ pi_n u_(n + 1) - pi_0 u_1 = pi_n u_(n + 1) - tau_1 $

and eventually...

$ pi_(N-1)/d_n = pi_N / b_N $

and finally...


$ tau_n = sum_(k=0)^(n-1) 1/pi_k sum_(j = k+1)^N pi_j / b_j $

== Example: SIS model

== Example: Branching process (Galtton-Watson)

#faint[N. Privault: Understanding Markov Chains]
#link("https://en.wikipedia.org/wiki/Galton%E2%80%93Watson_process")

This has an infinite number of states. Let's see what happens here.

We have probability $P[Y=r] = z_r$  such that $sum_(r=0)^oo z_r = 1$.

Also, $ P_(i j) & = P[x_n = i | x_(n-1) = j] \
        & = P[x_n = sum_(k=1)^j Y_r | x_(n-1) = j ] $

We are going to use... generating functions!

$ f(s) = sum_(r=0)^oo P(Y=r) s^r = E() $

If we change the variable we can find the generating function at generation $n$:

$
     F_n (s) & = sum_(r=0)^oo P[x_n = r] s^r \
             & = P(x_(n-1) = j) sum_(r=0)^oo P(x_n = r | x_(n-1) = j) s_j \
  P(x_n = r) & = P(x_n = r | x_(n-1) = j) P(x_(n-1) = j) \
$

What is the probability of the sum of $j$ random variables?

$
  sum_(r=0)^oo P(x_n = r | x_(n-1) = j) s_j = E(s^(sum_(=1)^j Y_k)) \
  =^"independent" E(s^Y)^j = f(s)^j
$

So the the term above can be written as:

$ F_n (s) = sum_(j=1)^oo P(x_(n-1) = j) f(s)^j = F_(n-1) (f(s)) $

and to put it more clearly:

$ F_n (s) = F_(n-1) (f(s)) $

And where does this recursion start? $F_0 (s) = s$ and $x_0 = 1$.

*Derivative:*

$ F'_n (s) = sum_(r=1)^oo P(x_n = r) r s^(r-1) $

Evaluated at one, we get the mean value at generation $m$:

$ F'_n (1) = E(x_n) = m_n $

The second derivative is:

$ F''_n (s) = sum_(r=1)^oo P(x_n = r) r (r-1) s^(r-2) $

And at $1$:

$ F''_n (1) = E(x_n^2) - E(x_n) $

This is almost the variance! In fact,

$ V_n = E(x_n^2) - E(x_n)^2 => F''_n (1) = V_n + m_n^2 - m_n $

#todo[A bunch of stuff missing.]

#todo[I woke up late, but...]

Properties #todo[of what? idk]
+ $1^T Z = 1^T$
+ $Z w = W$
+ $(I - P) Z = I - w 1^T$

We also get

$
  m_(i j) = (z_(i i) - z_(i j))/w_i
$

from the fundamental matrix.

== Continuous time Markov chains

We have that $P(t + u) = P(t) P(u)$.

There are events that happen that take a random, continuous amount of time. The waiting times are also called jump times and the inter-event times are also called holding times or sojourn times.

=== Example: Poisson process

This is a process where:
+ State space $cal(S) = { 0, 1, 2, ... }$
+ Stationary
+ Initially $X(0) = 0$
+ For sufficiently small $Delta t$:
  $
    P_(i j) (Delta t) = cases(
      lambda Delta t + o(Delta t),
      1 - lambda Delta t + o(Delta t),
      o(Delta),
      0
    )
  $

We eventually have to use generating function

$
  G(z, t) = sum_(i=0)^oo z^i P_i (t)
$

= Deterministic processes over averages or something

== Brownian motion

Observation of botonist (mr Brown) where a grain of polen in water is never stable and moves in an erratic way. He couldn't come up with an explanation of why it moved like that and it was ultimetaly used in one of the 1905 Einstein papers where it confirmed the material atom model.


If you take a lot of particles that have brownian motion you can have a macroscopic description of diffusion. This _is_ essentially deterministic, because for high enough number of particles the chance of that not happening is exponentially small, essentially 0.

=== Symmetric random walk: net displacement

How far away would a symmetric random walk move away from $0$ after some time?

We are going to say that the number of iterations is $m$, the position it is is $n$, the number of step to one direction is $n_1$ and in the other then is $n_2 = n - n_1$.

What is the probability that we are at position $n$ at time $m$? The answer is

$ P_n (m) = 1/2^n mat(n; n_1) $

which is the sum of $n$ Bernoulli variables.

Then, $m = n_1 - n_2 = 2n_1 - n$ so we can rewrite $n_1 = (m + n)/2$ and $n_1 = (m - n)/2$. This implies that $m$ and $n$ need to have the same parity. And the formula above becomes

$ P_n (m) = 1/2^n mat(n; n_1) = 1/2^n mat(n; (m+n)/2) $

#let mean(x) = $angle.l #x angle.r$

We can take averages and expected values. We want to calculate $mean(n_1)$. If we say that $p$ is the probability of going to the right and $q = 1 - p$ of going to the left, then

$ P_n (m) = mat(n; n_1) p^(n_1) q^(n - n_1) $

Then,

$ sum_(n_1=0)n P_n (m) = (p + q)^n = 1 $

So the expected value is

$
  mean(n_1) & = sum_(n_1 = 0)^n n_1 P_n (m) \
            & = sum_(n_1=0)^oo n_1 mat(n; n_1) p^(n_1) q^(n - n_1) \
            & = p partial/(partial p) (
                sum_(n_1=0)^oo mat(n; n_1) p^(n_1) q^(n - n_1)
              )
              #todo[what??] \
            & = p partial/(partial p) (p + q)^n \
            & = p n (p + q)^(n - 1) \
            & = n p \
$

We can do the same with $mean(n_1^2)$, and we get

$
  mean(n_1^2) & = n(n- 1) p^2 + n p \
              & = n^2 p^2 - n p^2 + n p \
              & = n^2 p^2 + n p (1 - p) \
$

And then we can calculate the variance:

$ v = mean(n_1^2) - mean(n_1)^2 $

*Average displacement after $n$ steps*

We calculate it this way:

$ mean(m) = mean(n_1) - mean(n_2) = n (p - q) $

Given $mean(n_1) = n p = mean(n_2)$ and $mean(n_1^2) = n^2 p^2 + n p q$. Then, the mean difference squared is:

$ mean(Delta n_1^2) = n p q $

Now, we can calculate the standard deviation:

$
  mean((Delta m)^2) = 4 n p q \
  sqrt(mean((Delta m)^2)) = 2 sqrt(n p q) \
$

If $p = q = 1/2$, then $mean(m) = 0$ which means no displacement, as expected. Also $sqrt(mean(Delta m^2)) = n^(1/2)$. This exponent is what characterizes brownian motion.


=== This is a Gaussian

$
  P_n (m) = 1/2^n (n!) / ( ((n+m)/2)! ((n-m)/2)! )
$

We use Stirling's approximation:

$
  n! approx sqrt(2 pi n) exp(n log n - n)
$

Substituting in:

$
  P_n (m) & = 1/2^n (sqrt(2 pi n)) / ( pi sqrt(n^2 - m^2) ) e^phi \
  "where" quad 1/n phi & = log n - 1/2 (1 + m/n) (log (n/2) + log (1 + m/n)) \
  & - 1/2 (1 - m/n) (log (n/2) + log (1 - m/n)) \
  & = log n - log(n/2) - 1/2 (1 + x) log (1 + x) - 1/2 (1 - x) log(1 - x) \
  & = log 2 - psi(x) \
  \
  \
  "where" quad psi(x) & = 1/2 (1 + x) log(1 + x) + 1/2 (1 - x) log(1 - x) \
  "and" quad x & = m/n \
$

So, $e^phi$ is now:

$
  e^phi = exp(n lg 2 - n psi(x)) = 2^n e^(-n psi(x))
$

Putting it al back in,

$
  P_n (m) & = sqrt((2n)/(pi (n^2 - m^2))) e^(-n psi(x)) \
          & = sqrt(2/(n pi)) (1 - x^2)^(-1/2) e^(-n psi(x)) \
$

Now, we use the fact that $x = m/n$ is generally very small, so we expand it in powers of $x$:

$
  sqrt(2/(n pi)) (1 - x^2)^(-1/2) = sqrt(2 / (n pi)) (1 + x^2/2 + ...)
$

and

$
  psi(x) & = 1/2 (1 + x) log(1 + x) + 1/2 (1 - x) log(1 - x) \
         & = x^2/2 + ... "(odd terms cancel)"
$


So, as $n$ tends to infinity, we find that

$
  P_n (m) = sqrt(2 / (n pi)) e^(-m^2 / (2n))
$

which is a gaussian.

== The diffusion equation

There is another way to calculate the above probability, which is closer to what we want to achieve.

We can talk about "number of paths", so

$ P_n (m) = 1/(2^n) ell(m, n) $

The number of paths, $ell$, can be determined recursively, since you can reach a position only from two positions before. That is,

$ ell(m, n) = ell(m - 1, n - 1) + ell(m + 1, n - 1) $

So now,

$ 2P_n (m) = P_(n-1) (m-1) + P_(n-1) (m + 1) $

And then we want to take macroscopic behavior so we use a large number of steps and pretend it is continuous:

$
  x = m Delta x \
  t = n Delta t \
  u(x, t) = (P_n (m)) / (Delta x)
$

And

$
  2 u(x, t) = u(x - Delta x, t - Delta t) + u(x + Delta x, t - Delta t)
$

Here we also do Taylor expansion

$
  u (x plus.minus Delta x, t - Delta t) &= u(x, t) plus.minus u_x (x, t) Delta x - u_t (x, t) Delta t + \
  & + 1/2 (U_(x x) (Delta x)^2 plus.minus 2 u_(x t) Delta x Delta t + u_(t t) (Delta t)^2 )
$

So,

$
  2 u(x, t) & = u(x - Delta x, t - Delta t) + u(x + Delta x, t - Delta t) \
  & 2u(x, t) - u_t(x, t) Delta t + u_(x x) (Delta x)^2/2 + u_(t t) (Delta t)^2/2 + ...
$

We set everything other than $2 u(x, t)$ t $0$, so we find

$
    0 & = -u_t Delta t + u_(x x) (Delta x)^2/2 + u_(t t) (Delta t)^2/2 + ... \
  u_t & = (Delta x)^2/(2 Delta t) u_(x x) + O(Delta t) = D u_(x x) + O(Delta t)
$

So if $D -> oo$ as $Delta x, Delta t -> 0$, then $u_(x x) = 0$, so $u = a + b x$, but since $u$ is bounded $u = a$ and since we start with $u = 0$ at the limits, then $u$ is _always_ $0$. Nothing ever happens, and I wonder.

And if $D -> oo$ then $u_t = 0$ so again it is a constant and is $0$.

Therefore, we need that $D$ is non-zero and finite. Then, we get the equation

$
  u_t = D u_(x x)
$

This is the diffusion equation.

==== Deriving the same equation as before

Using

$ n = t / (Delta t) quad m = x / (Delta x) $

we can find that

$
  u(x, t) = (P_n (m)) / (Delta x) = sqrt(2 /(pi n)) e^(-m^2/(2n)) 1 / (Delta x)
  = 1 / sqrt(pi D t) e^(-(x^2)/(4D t))
$

where $D = (Delta x)^2/(2 Delta t)$.

=== Concentration

The number of particles stays always the same at $N / sqrt(pi D t) exp(-x^2/(4 D t))$. We might care about the concentration f particles, which is

$
  C(x, t) = N/sqrt(4 D t pi) exp(-x^2/(4D t))
$

Properties of concentration:

+ $C(x, t)$ is a solution of the diffusion equation.
+ If $x != 0$, then $lim_(t -> 0^+) C(x, t) = 0$
+ $integral_(-oo)^oo C(x, t) dif x = N$

From this we derive that $C(x, 0) = N delta(x)$.

=== Fick's law

Another derivation of the difusion equation.

Imagine we have a region $Omega$ and a quantity $Q(t)$ that has a density $q(x, t)$ over $Omega$ such that

$ Q(t) = integral_Omega q(x, t) dif^n x $

Suppose there is also a current $J$ over the border

$
  dot(Q) = integral_Omega r(x, t) dif^n x - integral_(partial Omega) J dot hat(n) dif S
$

$
  dot(Q) = integral_Omega q_t (x t) dif^n x = integral_Omega sigma (x, t) dif^n x - integral_Omega gradient dot J dif^n x
$

So,

$ q_t = -gradient dot J + sigma $

Conservative systems have $sigma = 0$, with $q_t + gradient dot J = 0$. This is called the continuity equatin.

Non-conservative systems have $sigma != 0$

Any brownian motion has to satisfy $c_t = - gradient dot J$. If we can specify $J$ as a function of $J$ we can solve this.

So how do we find $J$? Suppose we have a distribution and the distribution is uneven. What tends to happen is that this uneveness tends to even out. That is, the loss should oppose the gradient of the concentration. So,

$ J = - D gradient c $

And thus,

$ c_t = gradient dot (D gradient c) $

We might have that $D$ is a function of $x$. For the case that this function is constant, we have the laplacian:

$ c_t = D gradient^2 c = D laplace c $

This is a nice result because it tells us in a very general sense that things tend to even out.

=== Solving the diffusion equation in an infinite domain

==== Fourier transform

We quickly need to introduce the Fourier transform. So, for a function $f : RR^n -> CC$ in $L^2$, the Fourier transform we use is

$
  cal(F) [f] (q) = hat(f)(q) = 1/(2pi)^(n/2) integral_(RR^n) f(x) e^(-i q x) dif^n x
$

And the inverse Fourier transform:

$
  cal(F)^(-1) [hat(f)] (q) = f(q) = 1/(2pi)^(n/2) integral_(RR^n) hat(f)(x) e^(i q x) dif^n x
$

*Properties:*

+ Linearity:
  $ cal(F) (alpha f + beta g) = alpha cal(F)(f) + beta cal(F)(g) $
+ Using the scaling operator $(Lambda_c f) (x) = f(c x)$
  $ cal(F)(Lambda_c f)(x) = Lambda_(c^(-1)) cal(F)(f) $
+ Translation to phase change:
  $ cal(T_a F) = e^(i q a) cal(F)0 $
+ $ cal(F)(gradient f) = i q cal(F)(f) $
+ Parseval's identity:
  $ integral_(RR^n) abs(f(x))^2 dif^n x = integral_RR abs(hat(f)(q))^2 dif q $
+ Convolution: Given
  $ (f convolve g)(x) = integral_(RR^n) f(y) g(x - y) dif^n y $
  we have
  $ cal(F) (f convolve g) = cal(F)(f) cal(F)(g) $


Let's take a look at the Fourier transform of the delta distribution, where

$
  delta_a (x) = delta(x - a), quad integral_RR^n delta_a (x) f(x) dif^n x = f(a)
$

$ cal(F) (delta_a)(q) = (e^(-i q a))/(2pi)^(n/2) $


The inverse Fourier transform of this is

$
  1/(2pi)^(n/2) integral_(RR^n) (e^(-i q a)) / (2pi)^(n/2) e^(i q x) dif^n x = delta(x - a)
$

This gives us a sort of orthogonality relation.

==== Solving diffusion with Fourier transforms

$ c_t = D laplace c $

Initially it can be any function, $c(x, 0) = f(x)$.

$
      hat(c)_t & = D abs(i q)^2 hat(c) = - D abs(q)^2 hat(c) \
  hat(c)(q, t) & = underbrace(hat(c)(q, 0), hat(f)(q)) e^(-laplace t abs(q))^2 \
               & = integral_(RR^n) f(y) g(x y, t) partial^n y
$

$
  cal(F)^(-1) (hat(g)) = g(x, t) & = 1/(2pi)^(n/2) integral_(RR^n) e^(-laplace t abs(q)^2 + i q x) dif^n q \
  & = 1/(2pi)^(n/2) integral_(RR^n) e^(-D t sum_j q_j^2 + i sum_j q_j x_j) dif^n q \
  & = product_(j=1)^n underbrace(
    (
      integral_(-oo)^oo 1/sqrt(2 pi) e^(-D t q_j^2 + i q_j x_j) dif q_j
    ), w(x_j, t)
  ) \
$

$
  w_x & = i/sqrt(2pi) integral_(-oo)^oo q e^(-D t q^2) e^(i q x) dif q \
  & = -i/(2 D t sqrt(2pi)) integral_(-oo)^oo (e^(-D t q^2))_q e^(i q x) dif q \
  & = [-i/(2 D t sqrt(2pi)) e^(-D t q^2 + i q x)]_(-oo)^oo + (i(i x))/(2 D t sqrt(2 pi)) underbrace(integral_(-oo)^oo e^(-D t q^2) e^(i q x) dif q, w)
$

So it results in

$
  w_x = (-x)/(2 D t) w \
  partial/(partial x) log w = -x/(2D t) \
  log((w(x, t)/w(0, t))) = -x^2/(4D t) \
  w(x, t) = w(0, t) e^(-x^2/(4D t)) \
$

where $ w(0, t) & = 1/sqrt(2pi) integral_(-oo)^oo e^(-D t q^2) dif q \
        & = 1/sqrt(2pi D t) integral_(-oo)^oo e^(-s^2) dif s = 1/sqrt(2 D t) $

So, $w(x, t) = 1/sqrt(2 D t) e^(-x^2/(4 D t))$

And finally,

$
  g(x, t) = 1/(2 gradient t)^(n/2) e^(abs(x)^2/(4D t))
$

== A couple of things I missed

#todo[]

== Fluid droplet formaton

#todo[]


== Pattern formation

We have the equation

$
  (partial h) / (partial t) = -nu partial_x^2 h - cal(K) partial_x^4 h
$

To solve this we go into frequency space:

$
  partial/(partial t) hat(h) = underbrace((v q^2 - k q^4), omega(q)) hat(h) \
  => h(x, t) = 1/sqrt(2pi) integral_(-oo)^oo hat(h)(q) e^(omega(q) t + dot(q) q x)
$

$
  h(x, t) = h_0 + epsilon(x, t) \
  epsilon(x, t) = epsilon(q) e^(omega(q)t + dot(q) q x)
$

=== Morphogenesis

From Alan Turing!

Things like zebra stripes happen because you have an activator and an inhibitor. The inhibitor inhibits the activator but the activator promotes itself _and the inhibitor_, you get a certain chain reaction that activates but at some point it starts reducing and after a certain diffusion it activates again and so on.

The equations are

$
  dot(u) = f(u, v) \
  dot(v) = g(u, v)
$

Where $u$ is the activator and $v$ is the inhibitor. We assume there is a stable fixed point $(u_0, v_0)$ such that $f(u_0, v_0) = g(u_0, v_0) = 0$.

We can write if $u_1 << u_0$ and $v_1 << v_0$

$
  u = u_0 + u_1 (x, t) \
  v = v_0 + v_1 (x, t)
$

We can rewrite this as

$
  dif/(dif t) underbrace(mat(u_1; v_1), W) = underbrace(mat(f_u, f_v; g_u, g_v), A) underbrace(mat(u_1; v_1), W) \
  (partial W) / (partial t) = A W - D (partial^2 W) / (partial x^2) \
  D = mat(D_0, 0; 0, D_v)
$

The solution of this is

$
  w = hat(w)(q) e^(omega t + i q x)
$

And eventually you get

$
  omega hat(w)(q) = underbrace((A - q^2 D), A(q)) hat(w) (q)
$

And if we calculate $det(A(q) - omega I)$, we get

$
  det(A(q) - omega I) = omega^2 - tr A(q) omega + det A(q) = 0
$

This is the dispersion relation.

The fact that $(u_0, v_0)$ is the stable point tells us that something about $A(0)$. Specifically, the trace and the determinant are on the stable region of the det/trace plot which means that
$
  det A(0) & = f_u g_v - f_v g_u > 0 \
   tr A(0) & = f_u + g_v < 0
$

Now let's look at $A(q)$ itself. The trace and determinants are

$
   tr A(q) & = underbrace(tr A(0), < 0) - q^2 underbrace((D_u + D_v), > 0) < 0 \
  det A(q) & = (f_u - q^2 D_u) (g_v - q^2 D_v) - f_v g_u \
           & = det A(0) - q^2 (f_u D_v + g_v D_u) + q^4 D_u D_v
$

The trace is always negative but the determinant isn't necessarily determined. But the stability is determined now by the determinant (pun not intended) so if $det A(q) > 0$ it is stable and if $det A(q) < 0$ it is unstable (and if $det A(q) = 0$ it is marginally stable). Also the eigenvalues follow, in each case:

- $det A(q) > 0 => Re(omega_(plus.minus)) < 0$ (stable)
- $det A(q) < 0 => Re(omega_-) < 0 < Re (omega_plus)$ (unstable)
- $det A(q) = 0 => Re(omega) = 0$ (marginally stable)

Let $q_m$ be the solutions of $det A(q) = 0$. We can calculate the value of $q_m$ as

$
  q^2_m = (f_u D_v + g_v D_u) / (2 D_u D_v) = 1/2 ((f_u)/(D_v) + (g_v)/(D_v)) > 0 \
  => (det A(0))/(D_u D_v) - underbrace(2q_m^2q_m^2 - q_m^4, -q_m^4) < 0 \
$

Wit h this we get the _Segel-Jacson's condition_:

$
  q_m^2 > sqrt((det A(0))/(D_u D_v))
$

Then, the conditions we have for instability is that

$
  cases(f_u + g_v < 0, (f_u)/(D_u) + (g_v)/(D_v) > 0)
$

The only way for this to happen is for one to be positive and one negative. Without loss of generallity, let's say $f_u > 0$ and $g_v < 0$. Then, we get from the first

$
  f_u + g_v < 0 => f_u < g_v => 1 < -(g_v)/(f_u) eq.triple r
$

And from the second:

$
  (f_u)/(D_u) > -(g_v) / (D v) => (D_v)/(D_u) > r
$

This is the more quantitative explanation of the qualitative behaviour we explained previously.

If we examine $A(0)$ and name it as

$
  A(0) = mat(-(b+1) + 2u v, u^2; b - 2u v, -u^2)lr(|, size: #250%)_(u_0, v_0) = mat(b-1, a^2; -b, -a^2)
$

We get that

$
  tr A(0) = b - 1 - a^2 < 0
  det A(0) = (1 - b) a^2 + a^2 b = a^2 > 0
$

which implies

$
  b < 1 + a^2
$

If we develop this in $q_m^2$ we get that

$
  (1 + a sqrt((D_u)/(D_v)))^2 < b < 1 + a^2
$

and

$
  q_c^2 = a / sqrt(D_u D_v)
$

== Rayleigh-Bénard convection

These are governed by Swift-Hohenberg equation

$
  (partial phi)/(partial t) = [epsilon - (1 + partial^2/(partial x^2))^2] phi + alpha phi^2 - phi^3
$

We are going to solve this.

First, we expand it:

$
  phi_t & = [epsilon - 1 - 2 partial^2_x - partial^4_x] phi + alpha phi^2 - phi^3 \
  & = -2 phi_(x x) - phi_(x x x x) + (epsilon - 1) phi - phi^3 + alpha phi^2`
$

We do a small perturbation

$
  phi_0 = 0 quad quad phi = phi_1 (x, t) quad #quote[small] \
  phi_1 = a(q) e^(i q x + omega t) \
  omega phi_1 = [epsilon -(1 - q^2)^2] phi_1 \
  omega = epsilon - (1 - q^2)^2 \
$

This is a dispersion relation. There is a critical value of $q$ where there is some instability.

These equations are simple enough where we can calculate the region of instability.

For $epsilon > 0$,

$
  1 - q^2 = plus.minus sqrt(epsilon) => q^2 = 1 plus.minus sqrt(epsilon) => q_plus.minus = sqrt(1 plus.minus sqrt(epsilon)) approx 1 plus.minus sqrt(epsilon)/2 \
  Delta q = q_+ - q_- = sqrt(epsilon)
$

(we always assume that $epsilon$ is small, whether positive or negative).

We are going to call $q_- (epsilon) < q < q_+ (epsilon) = B(epsilon)$ which is the unstable band.

Assimptotically, the solution is going to be governed by $phi_1$, so

$
  phi_1 (x, t) & ~ integral_(B(epsilon)) a(q, t) e^(i q x) dif q \
  "factor out dominant" quad & = e^(i q_c x) underbrace(integral_(B(epsilon)) a(q, t) e^(i (q - q_c) x) dif q, cal(A)(x, t))
  & = e^(i q_c x) cal(A)(x, t)
$

We are going to factor $Q = (q - q_c)/sqrt(epsilon)$ and so

$
  cal(A)(x, t) & = integral_(-1/2)^(1/2) tilde(a)(Q, t) e^(i Q sqrt(epsilon x)) dif Q \
  & = sqrt(epsilon) A(sqrt(epsilon) x, epsilon t)
  & = sqrt(epsilon) A(X, T)
$

Then,

$
  phi_1 (x, t) = e^(i q_c x) sqrt(epsilon) A(X, T)
$

(for small amounts of time).

Now,

$
  phi(x, t) = phi(x, X, T)
$

We are going to do perturbation analysis with multiple time-scales. So,

$
  partial_t -> epsilon partial_T \
  partial_x -> partial_x + sqrt(epsilon) partial_X
$

So then

$
  phi_t & = [epsilon - (1 + partial_x^2)^2] phi + alpha phi^2 - phi^3
  \ -> phi & = epsilon^(1/2) phi_1 + epsilon phi_2 + epsilon^(3/2) phi_3 + ... \
  & = epsilon^(1/2) psi quad "where" quad psi = phi_1 + epsilon^(1/2) phi_2 + ... \
  -> psi_t & = [epsilon - (1 + partial_x^2)^2]' psi + alpha epsilon^(1/2) psi^2 - epsilon psi^3 \
  -> epsilon psi_T & = [epsilon - (1 + partial_x^2 + 2 epsilon^(1/2) partial_x partial_X + epsilon partial_X^2)] psi + alpha epsilon^(1/2) psi^2 - epsilon psi^3
$

Here we need to do term by term analysis which is annoying so we introduce a couple of operators:

$
  L = 1 + partial_x^2 \
  D_(x X) = partial_x partial_X
  D_(X X) = partial_X^2
$

so,

$
  epsilon psi_T & = [epsilon - L^2 - 4 epsilon^(1/2) D_(x X) L - 4epsilon D_(x X)^2 - 2 epsilon D_(X X) L + ...] psi + alpha epsilon^(1/2) psi^2 - epsilon psi^3
$

Now we group terms. We get:

#let eps = $epsilon$

$
  epsilon psi_T & = epsilon phi_1 T + ... \
  epsilon psi & = epsilon phi_1 + ... \
  L^2 psi & = L^2 phi_1 + epsilon^(1/2) L^2 phi_2 + epsilon L^2 phi_3 + ... \
  4 eps^(1/2) D_(x X) L psi & = 4 eps^(1/2) D_(x X) L pi_1 + 4 eps D_(x X) L phi_2 + ... \
  4 eps D_(x X)^2 psi & = 4 eps D_(x X)^2 phi_1 + ... \
  2 eps D_(X X) L psi & = 2 eps D_(X X) L phi_1 + ... \
  alpha eps^(1/2) psi^2 & = alpha eps^(1/2) phi_1^2 + 2 alpha eps phi_1 phi_2 + ... \
  eps phi^3 & = eps phi_1^3 + ...
$

phew. Ok, so, for order $1$, we have $L^2 pi_1 = 0$ which is

$
  (partial^2/(partial x)^2 + 1)phi_1 = 0
$

we can solve this since it's homogeneous, and we get

$ phi_1 = A e^(i x) + overline(A) e^(-i x) $

We omit it for brevity, but all capital terms are assumed to be dependant on $X, T$. That is, $A = A(X, T)$.

For the terms of order $eps^(1/2)$, we have

$
  0 = -L^2 phi_2 - cancel(4 D_(x X) L phi_1) + alpha phi_1^2 \
  L^2 phi_2 = alpha phi_1^2 = alpha (A^2 e^(i 2x) + overline(A) e^(-i 2x) + 2 abs(A)^2) \
  phi_2 = B e^(i x) + overline(B) e^(-i x) + gamma + xi e^(i 2x) + overline(xi) e^(-i 2 x)
$

And now we have to calculate $L^2 phi_2$ 🤣🔫. Eventually, we get:

$
  gamma = 2 alpha abs(A)^2, xi = (alpha A^2)/9
$

So we get

$
  phi_2 = B e^(i x) + overline(B) e^(-i x) + 2 alpha abs(A)^2 + (alpha A^2)/9 e^(i 2x) + overline((alpha A^2)/9) e^(-i 2 x)
$

Finally, we have to do order $eps$. This is going to be even longer.

$
  phi_(1 T) & = phi_1 - L^2 phi_3 - 4 D_(x X) L phi_2 - 4 D_(x X)^2 phi_1 - 2 cancel(D_(X X) L phi_1) + 2 alpha phi_1 phi_2 - phi_1^3 \
  L^2 phi_3 & = phi_1 - phi_(1 T) - 4 D_(x X) L phi_2 - 4 D_(x X)^2 phi_1 + 2 alpha phi_1 phi_2 - phi_1^3
$

The homogeneous solution is of the type of $phi_1$. We have to kill the resonant terms. Fortunately, we can write it as the form:

$
  L^2 phi_3 = P e^(i x) + Q e^(i 2x) + R e^(i 3x) + "complx. conj."
$

We can deduce $P$ or something #todo[idk how really, but I also don't care really]

$
  P = A - A_T + 4A_(X X) + 4 alpha^2 A abs(A)^2 + 2 alpha^2 (A abs(A)^2)/9 - 3 A abs(A)^2
$

And since that is the amplitude of the resontant term, $P = 0$. So we can simplify

$
  0 = A - A_T + 4A_(X X) + 4 alpha^2 A abs(A)^2 + 2 alpha^2 (A abs(A)^2)/9 - 3 A abs(A)^2 \
  A_T = 4 A_(X X) + A - (3 - 38/9 alpha^2) A abs(A)^2
$

We call this the amplitude equation.

Here we do some things that I don't quite understand.

$
  (partial A)/(partial T) = 4 (partial^2 A)/(partial X^2) + A - (3 - 38/9 alpha^2) abs(A)^2 A \
  phi = eps^(1/2) [A e^(i x) + overline(A) e^(-i x)] + ... \
  (partial phi) / (partial t) = [eps - (1 + partial^2/(partial x^2))^2] phi + alpha phi^2 + phi^3
$

With a perturbation $phi(x+ x_0)$ we also conclude something I guess.

We can write the amplitude in a more general form

$
  (partial A)/(partial T) = gamma_1 (partial^2 A)/(partial X^2) + gamma_2 A + gamma_3 abs(A)^2 A \
  (partial overline(A))/(partial T) = overline(gamma_1) (partial^2 overline(A))/(partial X^2) + overline(gamma_2) overline(A) + overline(gamma_3) abs(A)^2 overline(A)
$

which implies that $gamma_1, gamma_2, gamma_3 in RR$.

And we can write it in a universal form, where

$
  (partial a)/(partial tau) = (partial^2 a)/(partial xi^2) + a - abs(a)^2 a
  "where" quad A = sqrt(gamma_2/gamma_3) a quad X = sqrt(gamma_1/gamma_2) xi quad T = tau/gamma_2 \
$

This equation is universal for any type I instability.

== Traveling waves

=== Fischer(-Kolmogorov-Petrovsky-Piscounov) equation

$
  partial_t A = k A(1 - A) + D partial_x^2 A
$

We want to look for travelling waves, so let's say $u(x, t) = U(z)$ where $z = x - c t$. The variable $c$ determines the direction of movement of the wave.

Then,

$
    u_t & = u_(x x) \
    u_t & = U'(-c) \
    u_x & = U' \
  -c U' & = U'' \
     U' & = -c a e^(-c z) \
      U & = a e^(-c z) + b
$

We are going to assume they're:
- bounded: $abs(U) < oo$
- and nonnegative: $U >= 0$

Then, $a$ needs to be $0$ to not decay exponentially, so we get $U = b$. This is not really a travelling wave, so we conclude these equations don't have travelling waves.

We need to modify the equation as so:

$
  -c U' = U'' + U (1 - U) \
  U'' + c U' + U(1 - U) = 0
$

This is the FK equation.

We are going to transform this into a system with
$
  cases(
    U' = V,
    V' = -c V - U(1 - U)
  )
$

We have fixed points $(0, 0)$ and $(1, 0)$ #todo[he's calling it "rest points" for some reason].

The jacobian of this is:

$
  Dif F = mat(0, 1; -1+2U, -c)
$

Assuming $c > 0$ (travelling waves going to the right):

$
    tau & = tr Dif F = -c < 0 \
  Delta & = det Dif F = 1 - 2 U
$

Since $Delta(0, 0) = 1 > 0$, $(0, 0)$ is a stable fixed point (either node or focus).

If $tau^2 >= 4Delta$ we get a node, if $tau^2 < 4A$ we get a focus. $tau^2 >= 4Delta$ gives the condition $c >= 2$.

For $(1, 0)$ we get $Delta(1, 0) = -1$ which is a saddle point.

There are some solutions (due to Kolmogorov). For instance, if we have

$
  u(x, 0) = cases(1 quad & x < x_1, 0 quad x > x_2) quad "where" x_1 < x_2
$

as initial condition, then we get a travelling wave of $c=2$, which is the minimum speed you can have (since for less, it is unstable).

We can see this by discarding the quadratic term:

$
  u_t = u_(x x) + u (1 - u) \
  -> u_t = u_(x x) + u
$

Then the solutions are

$
  U'' + c U' + U = 0 \
  U = A e^(-a z) \
  a^2 - c a + 1 = 0 \
$

where $c = (a^2 + 1)/a = a + 1/a$.

Then, we can have $a < 1$ or $a > 1$. This determines whether $e^(-a z) < e^(-z)$. Apparently, the solution needs to be bounded by $e^(-z)$ wch happens for $a < 1$ but for $a > 1$ you need to be stopped at $e^(-z)$ so then you get that, in fact, for $a > 1$ you get $c=2$, which is the minimum speed. This is not a proof, it's just an informal argument to understand it.

Now, if we have $u_t = u_(x x) + f(u)$ where $f$ has exactly two zeros and is positive otherwise, then we get the same behavior since qualitatively it's the same as the last example. Namely,

$ u = u_1 + eta quad quad eta_t = eta_(x x) + f'(u_1) eta $

where $eta$ is the thing we had before: $eta = U'' + c U' + f'(u_1) U = 0$.

There's also an asymptotic solution in $c$. We are going to take

$
  U(-oo) & = 1 \
   U(oo) & = 0 \
    U(0) & = 1/2 \
       0 & < eps = 1/c^2 <= 1/4 \
       c & >> 1
$

We need to scale the coordnate $c$, which we do with $xi = z/c = eps^(1/2) z$. Then,

$
  U'' + c U' + U(1 - U) = 0 \
  (dif U)/(dif z) = (dif U)/(dif xi) (dif x)/(dif z) \
  eps U'' + U' + U(1 - U) = 0 quad ("now" U' "is with respect to" xi)
$

Now,

$
  U(x) = g_0 (xi) + eps g_1(xi) + ... \
  eps (g''_0 + ...) + g'_0 + eps g_1' + ... + (g_0 + eps g_1 + ...) (1 - g_0 - eps g_1 - ...) = 0
$

- For order $1$, we have $g'_0 + g_0 (1 - g_0) = 0$.
- For order $eps$, we have $g''_0 + g'_1 (1 - g_0) g_1 - g_0 g_1 => g'_1 + (1 - 2g_0) g_1 = -g_0''$

To solve order $1$:

$
    g'_0 / (g_0 (1 - g_0)) & = -1 \
  1/g_0 + 1/(1 - g_0) g_0' & = -1 \
     (log(g_0/(1 - g_0)))' & = -1 \
      log(g_0 / (1 - g_0)) & = log A - xi \
           g_0 / (1 - g_0) & = A e^(-xi) \
                       g_0 & = A e^(-xi) / (1 + A e^(-xi)) \
                         A & = 1 quad ("using" g_0(0) = 1/2) \
                   g_0(xi) & = 1/(1 + e^xi)
$

To solve order $eps$, we can take the derivative of the order 1 solution:

$
              g'_1 + (1 - 2 g_0) g_1 & = -g''_0 \
             g''_0 + (1 - 2g_0) g'_0 & = 0 \
                            1 - 2g_0 & = -g''/g'_0 \
                 g'_1 - g''/g'_0 g-1 & = -g''_0 \
      (g'_1 g'_0 - g_1 g''_0) / g'_0 & = -g''_0 quad "almost quotient rule!" \
  (g'_1 g'_0 - g_1 g''_0) / (g'_0)^2 & = -g''/g'_0 \
                       (g_1 / g'_0)' & = -g''_0 / g'_0 = -(log abs(g'_0))' \
         (g_1/g'_0 + log abs(g_0'))' & = 0 \
            g_1/g'_0 + log abs(g'_0) & = B \
                                 g_1 & = B g'_0 - g'_0 log abs(g'_0) \
                                   0 & = B g'_0 (0) - g'_0 log abs(g'_0 (0)) \
                                   B & = log abs(g'_0 (0)) \
                            g'_0(xi) & = - e^xi/(1 + e^xi)^2 \
                             g'_0(0) & = 1/4 \
                                 g_1 & = -g'_0 log 4 - g'_0 log abs(g'_0) \
                                 g_1 & = -g'_0 log abs(4 g'_0)
$

So, finally,

$
  g_1 & = - e^xi/(1 + e^xi)^2 log ((4e^xi)/(1 + e^xi)^2)
$

And then,

$
  U(z) = g_0 (z/c) + 1/c^2 g_1 (z/c) + ...
$

We assumed $c$ is very big, but even if we take $c=2$ which is the worst case scenario, it is still accurate within a few percent. It's a reasonable solution.

=== Multispecies systems and predator/prey coexistence

$
  (partial N_1)/(partial t) = A N_1 (1 - N_1/K) - B N_1 N_2 + D_1 (partial^2 N_1)/(partial x^2) \
  (partial N_2)/(partial t) = C N_1 N_2 - D N_2 + D_2 (partial^2 N_2)/(partial x^2)
$

$
  u = N_1 / K quad quad v = (B N_2)/A
$

$ t' = A t quad quad x' = x (A/D_2)^(1/2) $

$
  D' = D_1/D_2 quad a = (C K)/A quad b = D/(C K)
$

$
  (partial u)/(partial t) & = u (1 - u - v) + D (partial^2 u)/(partial x^2) \
  (partial v)/(partial t) & = a v (u - b) + (partial^2 v)/(partial x^2)
$

We assume that $u, v >= 0$ since they represent concentrations.

Uniform solution is

$
  u (1 - u - v) = 0 \
  v (u - b) = 0
$

this has solutions $u = 0$ and $u + v = 1$, or $v = 0$ and $u = b$. So the fixed points are $(0, 0)$ for exinctions, $(1, 0)$ is predator extintion and $(b, 1 - b)$ is coexistence, for $b < 1$. We are going to focus on this case.

To check what kind of points these are, we calculate the jacobian and whatnot:

$
        Dif F & = mat(1 - 2u - v, -u; a v, a(u - b)) \
  Dif F(0, 0) & = mat(1, 0; 0, -a b) \
  Dif F(1, 0) & = mat(-1, -1; 0, a (1 - b)) \
$

Therefore $(0, 0)$ and $(1, 0)$ are saddles.

$
  Dif F(b, 1 - b) = mat(-b, -b; a (1 - b), 0)
$

Characteristic equation is $lambda^2 + lambda b$

Trace is $tau = -b < 0$, determinant is $Delta = a b (1 b) > 0$. Therefore these points are attractors.

It is a node if

$
  tau^2 & >= 4 Delta \
    b^2 & >= 4 a b(1 - b) \
      b & >= 4 a (1 - b) \
      a & <= b/(4(1 - b)) \
$

It is a focus if

$
  tau^2 & < 4 Delta \
      a & > b/(4(1-b))
$

So now, let's find travelling waves, using

$
    u(x, t) & = U(z) \
    v(x, t) & = V(z) \
  "where" z & = x + c t
$

The equations we now get are

$
  cases(
    C U' & = V (1 - U - V) + D U'' \
    C V' & = a V (U - b) + V'' \
  )
$

We are going to have $D_1 << D_2$, $D << 1$ and actually just $D = 0$. So we get

$
  cases(
    U' = 1/C U (1 - U - V) \
    V' = W \
    W' = C W - a V(U - b)
  )
$

We are going to look at cases where we start at $(0, 0, 0)$ or $(1, 0, 0)$ and end at the coexistence region $(b, 1 - b, 0)$. This gives boundaries conditions

$
  U(-oo) = 1 quad U(oo) = b \
  V(-oo) = 0 quad V(oo) = 1 - b
$

in one case, and

$
  U(-oo) = 0 quad U(oo) = b \
  V(-oo) = 0 quad V(oo) = 1 - b
$

for the other. We are going to focus on the first one because it's more interesting but the second one can be analyzed in the same way.

$
           Dif F & = mat(1/C(1 - 2U - V), -1/C U, 0; 0, 0, 1; -a V, -a(U - b), c) \
  Dif F(1, 0, 0) & = mat(-1/C, -1/C, 0; 0, 0, 1; 0, -a (1 - b), c)
$

The matrix is block diagonal, so one eigenvalue is $lambda_0 = -1/c$ and the other two are $lambda_(plus.minus) = (c plus.minus sqrt(c^2 - 4 a (1- b)))/2$. If the eigenvalues are real (i.e., $c^2 >= 4 a(a - b)$) then the square root is smaller than $c$ and so the eigenvalues are real. The other case doesn't happen because it would imply negative concentrations.

Therefore, the point is a saddle-focus with $c_"min" = 2 sqrt(a(1-b))$.

$
  Dif F (b, 1 - b, 0) = mat(-b/C, -b/C, 0; 0, 0, 1; -a(1 - b), 0, c)
$

The characteristic polynomial turns out to be

$
  lambda^3 - (C - b/C) lambda^2 - b lambda - (a b (1 - b))/c = p(lambda)
$

This cannot be simplified, so we're screwed 🥺. But we can put $a = 0$ to at least see the shape of the polynomial or something.

$
  lambda (lambda^2 - (C - b/C) lambda - b) & = 0 \
  lambda_0 & = 0 \
  lambda_(1, 2) & = 1/2 (C - b/C plus.minus sqrt((C - b/C)^2 + 4b)) \
  & = 1/(2C) (C^2 - b plus.minus sqrt((C^2-b)^2 + 4b C^2) \
$

We can also calculate the extrema:

$
           p'(lambda) & = 3 lambda^2 - 2(C - b/C) lambda - b \
  lambda_(plus.minus) & = 1/6 (2 (C - b/C) plus.minus sqrt(4(C - b/C)^2) + 12 b) \
                      & = 1/(3C) (C^2 - b plus.minus sqrt((C^2 - b)^2 + 3 b C^2)
$

=== Travelling waves that are actually sine waves that are moving

The equations are

$
  (partial u)/(partial t) = lambda (a)u omega (a) v + (partial^2 u)/(partial x^2) \
  (partial v)/(partial t) = lambda(a) v + omega(a) u + (partial^2 v)/(partial x^2) \
$

$
  a = sqrt(u^2 + v^2) \
  z = u + i v \
  lambda + i omega
$

$
  (partial z)/(partial t) = (lambda(a) + i omega (alpha)) (partial^2 z)/(partial x^2) \
  z = a e^(i phi) \
  a_t/a z + i phi_t z = (lambda(a) + i omega(a)) z + (a_(x x)/a - phi_x^2 + i(phi_(x x) + 2 a_x/a phi_x)) z \
  a_t/a = lambda(a) + a_(x x)/a - phi_x^2 \
$

Here we have to calculate $z_x$ and $z_(x x)$.

Finally,

$
    a_t & = a lambda(omega) a_(x x) - a phi_x^2 \
  phi_t & = omega(a) + phi_(x x) + 2 phi_x a_x/a
$

There is a limit cycle at $omega(gamma)$ where $z = z_0 e^(i omega(gamma) t)$ and $z_0$ is an arbitrary phase $z_0 = gamma e^(nu phi_0)$

*Finding the travelling wave*

$
  a = alpha \
  phi = 6t - q x\
$

$
  0 = alpha lambda(alpha) - a q^2 \
  q = sqrt(lambda(a)) \
  sigma = omega(a)
$

$
  cases(
    u = alpha cos(omega(alpha) t - sqrt(lambda(a)) x),
    v = alpha sin(omega(alpha) t - sqrt(lambda(a)) x),
  ) \
             c & = omega(a) / sqrt(lambda(alpha)) \
  "wavelength" & = 2pi/sqrt(lambda(a))
$

for $alpha < gamma$. The speed increases up to infinity and the wavelength diverges too so the whole system jumps to the limit cycle.

