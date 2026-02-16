#import "../format.typ": *

#show: notes(
  subtitle: [Spectral Graph Theory],
)[Applied Discrete Mathematics]

= Combinatorics

The main goal is to count the elements of a set without enumerating them. Sets are unordered, cardinality is the number of distinct elements in a set. If two sets have a bijection they have the same cardinality. Generally we do bijections to a subset of the natural numbers to count them.

#example[
  Let $S = { n in NN : n | 6000 }$.

  Given that $6000 = 2^4 dot 3^1 dot 5^3$ then we know that all elements are of the form $2^alpha 3^beta 5^gamma$, so the solution are just $5 dot 2 dot 4 = 40$.
]

