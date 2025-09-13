#import "../format.typ": *

#show: exercises[1]

#exercise[
  When a drop of liquid hits a wetted surface a crown formation appears. The number of points $N$ on the crown has been found to depend on the speed $U$ at which the drop hits the surface, the radius $r$ and density $rho$ of the drop, and the surface tension $sigma$ of the liquid making up the drop.

  #exercise[
    Use dimensional reduction to determine the functional dependence of the number $N$ of points on $U$, $r$, $ρ$, and $σ$. Express your answer in terms of the Weber number $W_e = ρ U^2 r/σ$.

    Note: The dimensions of surface tension are force per unit length or energy per unit surface.
  ][
    Put the units for the Weber number:

    $
      [W_e] & = [rho] [U]^2 [r] [sigma]^(-1) \
            & = (M L^(-3)) (L T^(-1))^2 (L) (M L T^(-2) L^(-1))^(-1) \
            & = 1
    $

    And for the main equation:

    $
        N & = f(U, r, rho, sigma) \
      [N] & = [U]^a [r]^b [rho]^c [sigma]^d \
        1 & = (L T^(-1))^a L^b (M L^(-3))^c (M cancel(L) T^(-2) cancel(L^(-1)))^d \
          & = L^a T^(-a) L^b M^c L^(-3c) M^d T^(-2d) \
          & = M^(c + d) L^(a + b - 3c) T^(-a - 2d) \
    $

    We get the following system:

    $
      cases(
        c + d = 0,
        a + b - 3c = 0,
        -a -2d = 0,
      )
    $

    Where we get $a = -2d$, $b = -d$, $c = -d$. So:

    $
      N & = U^(-2d) r^(-d) rho^(-d) sigma^d \
        & = (sigma / (rho r U^2))^d
    $

    Since the quantity is dimensionless, we only know it's an function of this quantity. Therefore, in terms of $W_e$ we have:

    $ N = f(1 / W_e) $
  ]





  #exercise[
    The value of N has been measured as a function of the initial height $h$ from which the drop is released and the results are shown in panel (b) of the figure. Express your answer to part 1.1 in terms of h by computing $U$ in terms of $h$ and the acceleration of gravity $g$. Assume the drop starts with zero velocity and use the projectile problem within the approximation of a uniform acceleration
  ][
    The speed of a projectile after falling for a height $h$ can be computed as $g h$. So $U = g h$. Rewriting $W_e$ with this we get $W_e = h^2 rho g^2 r / sigma$ and so

    $ N = f(1 / W_e) = f(sigma / (h^2 rho g^2 r)) $
  ]

  #exercise[
    The data in panel (b) of the figure show that N is negligible for small h, but once the height is large enough, then N grows linearly with h. Use this, and your result from part 1.2, to find the unknown function in part 1.1. In the experiments, r = 3.6 mm, ρ = 1.1 g/cm3 , and σ = 50.5 dyn/cm.

    Note: dyn is the unit of force in the CGS system which uses cm, grams (g), and seconds as fundamental units.
  ][
    Firstly, we need the value of gravity $g$ which I'll take as $9.81 "m"/"s" = 981 "cm"/"s"$.

    Then, we have $N$ and we know experimentally that it has a slope of about $1/4$ with respect to $h$ so we know

    $ N prop h/2 $

    We can calculate what are the values we get. $1/W_e$ for this data is:

    $ 1/W_e = 50.5 / (h^2 1.1 dot 981^2 dot 0.36) = 0.0001325129 1/h^2 $

    Let's call the constant $0.0001325129 = k_"small"$.

    And we know that, for these values, $f(1 / W_e) = h / 4$ (since the slope is $1/4$), so we can work backwards from that. Firstly, we know we need a square root and to invert, and then we need to do some scaling. Therefore, our function has to be:

    $ f(x) = alpha 1/sqrt(x) $

    As for the alpha, we can compute it in terms of $k_"small"$:

    $
      f(1/W_e) = alpha 1 / sqrt(1 / W_e) = alpha 1 / sqrt(k_"small" 1 / h^2) = h alpha / sqrt(k_"small")
    $

    And since we know $f(1/W_e) = h/2$ we have $cancel(h)/4 = cancel(h) alpha / sqrt(k_"small")$ therefore $alpha = sqrt(k_"small") / 4 approx 0.0028778563$

    Thus, our function is:

    $ N = alpha sqrt(W_e) = 0.002878 sqrt(W_e) = h/4 $
  ]

  #exercise[
    According to your result from part 1.3, what must the initial height of the drop be to produce at least $N = 80$ points?
  ][
    Simply $80 = h/4 => h = 320$. This doesn't really need the results from part 1.3, it just needs the results from the graph.
  ]
][]
