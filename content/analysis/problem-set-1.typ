#import "../format.typ": *

#show: exercises[]

#exercise[
  Let $X$ be the set of students in this master’s course, and let $d : X × X → [0, ∞)$
  be defined by
  $
    d(x, y) = cases(
      0\, space x = y,
      1\, space x != y
    )
  $

  Is $(X, d)$ a metric space?
][
  Yes, it is:
]

#exercise[
  Let X be the set of all polynomials (of any degree) with coefficients a0, . . . , an ∈ {0, 1}. Prove that the set A = {p(1/2) : p ∈ X} is dense in B = [0, 2].
][

]

#exercise-counter.update(8)

#exercise[
  Let $f(x) = e^x$. Give explicitly the polynomial of degree 2 that best approximates $f$ with respect to the norm

  $ norm(g) = (integral_(-infinity)^infinity |g(x)|^2 e^(-x^2) dif x)^(1/2) $

  In other words: for which polynomial p of degree 2 is ∥f − p∥ minimal? What is the minimal error?
][

]
