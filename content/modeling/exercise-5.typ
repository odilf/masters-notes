= Modeling: Assignment 5

_By Odysseas Machairas_

$
  partial_t u = D partial_x^2 u - c partial_x u, quad "for" cases(-oo < x < oo, 0 < t)
  \ \ u(x,0) = f(x), D > 0
$

== 5.1

Taking the Fourier transform of $u$, $cal(F)[u](x) = hat(u) (q)$, using properties of the Fourier transform we can write:

$
          partial_t u & = D partial_x^2 u - c partial_x u \
  cal(F)(partial_t u) & = cal(F)(D partial_x^2 u - c partial_x u) \
     partial_t hat(u) & = D (i q)^2 hat(u) - c (i q) hat(u) \
     partial_t hat(u) & = (-D q^2 - c i q) hat(u) \
         hat(u)(q, t) & = A e^(-t(D q^2 + c i q)) \
$

The initial condition is given by $hat(u)(q, 0) = cal(F)[f](x) = hat(f)(q)$, so

$ hat(u)(q, t) = hat(f)(q) e^(-t(D q^2 + c i q)) $

To move back to the space domain, we have to take the inverse fourier transform:

$
  u(x, t) & = 1/sqrt(2pi) integral_(-oo)^oo hat(u)(q, t) e^(i q x) dif q \
  & = 1/sqrt(2pi) integral_(-oo)^oo hat(f)(q) e^(-t(D q^2 + i q)) e^(i q x) dif q \
  & = 1/(2pi) integral_(-oo)^oo integral_(-oo)^oo f(y) e^(-i q y) dif y space e^(-t(D q^2 + i q)) e^(i q x) dif q \
  & = 1/(2pi) integral_(-oo)^oo f(y) integral_(-oo)^oo e^(-i q y) e^(-t(D q^2 + i q)) e^(i q x) dif q dif y \
  & = integral_(-oo)^oo f(y) 1/(2pi) integral_(-oo)^oo e^(-t D q^2 + i q(x - c t - y)) dif q dif y \
  & = integral_(-oo)^oo f(y) 1/sqrt(4 pi D t) e^(-(x - c t - y)^2/(4D t)) dif y \
  & = (f convolve G(dot, t)) (x - c t) \
  "where" G(x, t) & = 1/sqrt(4 pi D t) e^(-x^2/(4D t))
$

We used the fact that $G$ is the heat kernel which has Fourier transform $e^(-D k^2 t)$ so we can see that we're just taking the inverse Fourier transform of the Fourier transform of $G$ (it helps that we have an expectation of what the solution will look like). However, for completeness, we can also show this by completing the square:

$
  1/(2pi) integral_(-oo)^oo e^(-t D q^2 + i q x ) dif q
  &= 1/(2pi) integral_(-oo)^oo exp[-t D (q^2 - (i x )/(t D) q)] dif q \
  &= 1/(2pi) integral_(-oo)^oo exp[-t D (q - (i x )/(2 t D))^2 - ( x^2)/(4 D t)] dif q \
  &= 1/(2pi) e^(-( x^2)/(4 D t)) integral_(-oo)^oo exp[-t D (q - (i x )/(2 t D))^2] dif q \
  &= 1/(2pi) e^(-( x^2)/(4 D t)) integral_(-oo)^oo e^(-t D u^2) dif u quad quad "(substitute " u = q - (i x )/(2 t D)")"\
  &= 1/(2pi) sqrt(pi/(t D)) e^(-( x^2)/(4 D t)) quad quad "(Gaussian integral)"\
  &= 1/sqrt(4 pi t D) e^(-( x^2)/(4 D t)) \
$

== 5.2

Taking $xi = x - c t$, $tau = t$ and $v(xi, tau) = u(x, t)$ we have that

$
  partial_tau v = partial_t u + c partial_x u \
  partial_xi v = partial_x u
$

The original equation, then, becomes

$
                       partial_t u & = D partial_x^2 u - c partial_x u \
  -> partial_tau v - c partial_x u & = D partial_xi v - c partial_x u \
                    => partial_t v & = D partial_x^2 v
$

which is indeed the diffusion equation. The diffusion equation has solution

$
  v(xi, tau) & = 1/sqrt(4pi D tau) integral_(-oo)^oo f(s) e^(-(xi-s)^2/(4 D tau)) dif s \
  => u(x, t) & = 1/sqrt(4pi D t) integral_(-oo)^oo f(s) e^(-(x - c t - s)^2/(4 D t)) dif s
$

which is the solution we arrived at previously.

== 5.3

The physical interpretation is that $c$ is a speed at which the substance in question is moving at. It is not that clear from the original equation but it is very clear with $xi = x - c t$. The substance moves at a speed of $c$ units of space per unit of time and the $xi = x - c t$ is a frame of reference change.

Particularly, if $c > 0$ the center of the diffusion is going to be at $x - c t = 0 => x = c t$, so it is moving to the right. Conversely, if $c < 0$ then the center of diffusion is moving to the left.

And, to complete the physical interpretation, while $c$ is the velocity of the movement of the substance, $D$ is the rate of diffusion.

== 5.4

We have equation

$
  partial_t u = D partial_x^2 u - c u
$

For this equation we have a term of diffusion but we have another term that makes the derivative decrease proportionally to the current concentration. $c$ is constant in space so it corresponds to some sort of exponential decay. We could see this for the diffusion of an element where the half-life is short enough compared to the timescale of interest where we have to take into account the radioactive decay, which is this sort of exponential decay.

We can solve it again by taking the Fourier transform and solving in in the frequency domain:

$
  frac(partial hat(u), partial t) & = -D k^2 hat(u) - c hat(u) \
                                  & = -(D k^2 + c) hat(u) \
                   => hat(u)(k,t) & = hat(f)(k) e^(-(D k^2 + c)t)
$

then taking the inverse transform as before:

$
  u(x,t) & = 1/(2pi) integral_(-oo)^(oo) hat(f)(q) e^(-D q^2 t) e^(-c t) e^(i q x) dif q \
  & = e^(-c t) dot 1/(2pi) integral_(-oo)^(oo) hat(f)(q) e^(-D q^2 t) e^(i q x) dif q
$

Here we are again taking the inverse Fourier transform of the Fourier transform of the heat kernel $G(x, t)$ multiplied by $hat(f)$, and since the calculation before is very similar we can be more brief and cleverer:

$
  u(x, t) & = e^(-c t) cal(F^(-1)) (hat(f) dot cal(F)(G(dot, t))) \
  & = e^(-c t) (f convolve G(dot, t)) \
  & = e^(-c t)/(sqrt(4pi D t)) integral_(-infinity)^(infinity) f(s) e^(-(x-s)^2/(4D t)) dif s \
$

== 5.5

From the previous exercise it's pretty clear that $a = -c$, so that $u = v e^(-c t)$. This means that $v = u e^(c t)$ and thus

$
  partial_t u = partial/(partial t) v e^(-c t) = partial_t v e^(-c t) - c v e^(-c t) \
  partial_x u = partial/(partial x) v e^(-c t) = e^(-c t) partial_x v \
  partial_x^2 u = partial/(partial x) e^(-c t) partial_x v = e^(-c t) partial_x^2 v \
$

Then, the equation to solve becomes:

$
                          partial_t u & = D partial_x^2 u - c u \
  partial_t v e^(-c t) - c v e^(-c t) & = D e^(-c t) partial_x^2 v - c v e^(-c t) \
                 partial_t v e^(-c t) & = D e^(-c t) partial_x^2 v \
                          partial_t v & = D partial_x^2 v \
$

Once again we have the standard diffusion equation, with the solution equivalent to the previous exercise:

$
  v(x, t) & = 1/sqrt(4pi D t) integral_(-oo)^oo f(s) e^(-(x-s)^2/(4 D t)) dif s \
  => u(x, t) & = e^(-c t) 1/sqrt(4pi D t) integral_(-oo)^oo f(s) e^(-(x-s)^2/(4 D t)) dif s \
$

== 5.6

This is a similar case as the one in 5.4, except that now it's exponential growth instead of decay. This is common for organisms that reproduce, so we could see it in the diffusion of bacteria (as long as they haven't reached the carrying capacity of the environment). Another example that is perhaps more fun is that this could be used to model the "diffusion" of a species (like humans), throughout the globe, again assuming that the carrying capacity isn't reached, and probably assuming a lot of other unreallistic things, so let's keep the bacteria example :)
