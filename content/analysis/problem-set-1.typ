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
