#import "@preview/h-graph:0.1.0": (
  enable-graph-in-raw, polar-render, tree-render,
);

#import "../format.typ": *

#show: notes()[Applied Discrete Mathematics]


I have only made notes for the last part of the course, we assume some prior knowledge and notation.

= Spectral Graph Theory

== Definitions and notation

#let origin = $partial_-$
#let endpoint = $partial_+$
#definition[Directed pseudograph][
  A _directed pseudograph_ is a tuple $G = (V, E) = (V, E, partial)$ where:
  - $V$ is the set of vertices
  - $E subset (V times V)$ is the set of (oriented) edges

  $partial: E -> V times V$ is the _incidence map_, defined as that for an edge $a b in E$, $ partial(a b) = (a, b). $

  We also define auxiliaries $origin$ and $endpoint$ where

  $
    partial(e) = (origin(e), endpoint(e))
  $
]

Observation: we can convert any undirected graph to a directed pseudograph by replacing every undirected edge by two directed edges.

#definition[Edge sets of vertices][
  For each edge we define the following edges sets:
  - $E_v^+ = { e in E : endpoint e = v }$: incoming edges to $v$
  - $E_v^- = { e in E : origin e = v }$: outgoing edges of $v$
  - $E_v = E_v^- union.sq E_v^+$: incident edges of $v$
]

#definition[Vertex degrees][
  - The _out-degree_ of $v$ is $delta_- (v) = abs(E_v^-)$
  - The _in-degree_ of $v$ is $delta_+ (v) = abs(E_v^+)$
  - The _degree_ of $v$ is $delta(v) = abs(E_v) = delta_+ (v) + delta_- (v)$
]

#definition[Edge sets of sets of vertices][
  - $E^+ (A, B) = { e in E : origin e in A, endpoint e in B }$: Edges that start at $A$ and end at $B$.
  - $E^- (A, B) = { e in E : endpoint in A, origin e in B }$: Edges starting at $B$ and ending at $A$.
  - $E(A, B) = E^-(A, B) union E^+(A, B)$.

  For singleton sets, we can write the shorthand $E(A, { v }) = E(A, v)$.
]

Note: for edge sets of single vertices we take _disjoint_ union, for edge sets of sets of vertices we take _regular_ union. That means that $E_v$ counts self-loops as $2$. But if $A$ and $B$ has an edge that goes from $A$ to $B$ and also from $B$ to $A$, which can only happen if $A$ and $B$ share an vertex that has a self loop, which only counts for one.

#lemma[
  - For all $v in V$, we have that $E_v^plus.minus = E^plus.minus(V, { v }) = E^plus.minus(V, v)$.
  - $E^+ (A, B) = E^- (B, A)$.
]

We have no restriction on $V$ to be finite, but it has to be...

#definition[Locally finite][
  A graph $G = (V, E)$ is _locally finite_ if $ forall v in V: delta(v) < oo $
]

#lemma[
  If $abs(V) = n$ is finite, then $abs(E) = 1/2 sum_(v in V) delta(v) < oo$ (by handshake lemma).
]

#definition[Adjecency and incidence matrix][
  The _adjacency matrix_ $A^G$ of a graph $G = (V, E)$ is an $abs(V) times abs(V)$ matrix defined as

  $
    A_(j k)^G = abs(E_+ (v_j, v_k)) + abs(E_- (v_j, v_k)) = cases(
      2 times ("# loops at" v_j) quad & "if" j = k,
      abs(E(v_j, v_k)) & "if" j != k
    )
  $

  The _incidence matrix_ $B^G$ of a graph $G$ is an $abs(V) times abs(E)$ matrix defined by:

  $
    B_(i j)^G = cases(
      1 quad & "if" endpoint e_j = v_i and origin e_j != v_i,
      -1 quad & "if" origin e_j = v_i and endpoint e_j != v_i,
      0 quad & "otherwise",
    )
  $
]

In other words, the adjecency matrix counts the number of edges between any two vertices, and the incidence matrix has a column vector with a $1$ and $-1$ for each vertex that an edge connects (except for self-loops).

== Weighted graphs, derivative and adjoint

#definition[Weighted graph][
  A weighted graph is a tuple $(G, m)$ where $G = (V, E)$ is an oriented pseudograph and $m$ is a set of two functions

  $
    m: V -> (0, oo) quad "and" quad m: E -> (0, oo)
  $

  so, yes, we are defining two functions via the same symbol...
]

#definition[Measure spaces #todo[is this rigorous?]][
  Given a weighted graph $(G, m)$, we can restrict functions that act on vertices or edges of our graph to an equivalent of an $ell_p$ space (in practice, almost always $ell_2$) as:

  $
    ell_p (V, m)= { phi : V -> CC : sum_(v in V) abs(phi(v))^p m(v) < oo } \
    ell_p (E, m)= { eta : E -> CC : sum_(e in E) abs(eta(e))^p m(e) < oo }
  $

  These are endowed with the corresponding inner product for some function $phi, psi: V -> CC$ and $eta, xi : E -> CC$:

  $
    innerproduct(phi, psi)_(ell_2(V, m)) = sum_(v in V) phi(v) overline(psi(v)) m(v), quad norm(phi)^2_(ell_2 (V, m)) = innerproduct(phi, phi)_(ell_2 (V, m)) \
    innerproduct(eta, xi)_(ell_2(E, m)) = sum_(e in E) eta(e) overline(xi(e)) m(e), quad norm(eta)^2_(ell_2 (E, m)) = innerproduct(eta, eta)_(ell_2 (E, m))
  $
]

#definition[Intrinsic weights][
  There are two weights that are particularly important:
  + _Combinatorial weight_: $m(v) = 1$, $m(e) = 1$
  + _Standard weight_: $m(v) = delta(v)$, $m(e) = 1$

  We can denote with shorthand a graph $G$ with one of these weights as $G^"comb"$ and $G^"std"$.

  We can also shorthand $ell_2(V, 1) = ell_2(V)$ for the combinatorial weight.
]

Note that $ell_2(V, 1) tilde CC^abs(V)$ and $ell_2 (E, 1) tilde CC^abs(E)$ (by writing a vector of each value of $phi$ for each vertex) #todo[does this not happen with other weights?].

#let rel = "rel"

#definition[Relative weight][
  Given a weight $m$, we can calculate its _relative weight_ as

  $
    rel_m : V -> (0, oo) \
    rel_m (v) = 1/m(v) sum_(e in E_v) m(e)
  $

  A weight is said to be _normalized_ if $rel_m (v) = 1$ for all $v in V$.
]

#example[Are intrinsic weights normalized?][
  For combinatorial, $rel_"comb" (v) = 1/1 sum_(e in E) 1 = delta(v)$ which is not necessarily $1$.

  For standard weights, $rel_"std" (v) = 1/delta(v) sum_(e in E) 1 = 1$, it is normalized!

  In fact, combinatorial weight is only normalized if all vertices have $delta(v) = 1$, which makes it just be the same as the standard weight.
]

#definition[Discrete derivative operator and adjoint][
  The _discrete derivative_ $dif: ell_2 (V, m) -> ell_2 (E, m)$ is defined as

  $
    (dif phi) (e) = phi(endpoint e) - phi(origin e)
  $

  that is, as the difference of the function evaluated at each side of an edge.

  The _adjoint_ of the derivative operator $dif^*: ell_2(E, m) -> ell_2(V, m)$ is the discrete analogue to the _divergence_ and is defined as the only operator that satisfies

  $
    innerproduct(dif phi, eta) = innerproduct(e, dif^* eta).
  $
]

#theorem[  The adjoint $dif^*$ of a function $eta: ell_2 (E, m)$ is given explicitly by

  $
    (dif^* eta)(v) = 1/m(v) sum_(e in E_v) arrow(eta_e)(v) m(e)
  $

  where $ arrow(eta_e) = cases(eta(e) quad & "if" v = endpoint e, -eta(e) quad & "if" v = origin e). $

  Equivalently,

  $
    (dif^* eta) (v) = 1/m(v) (sum_(e in E_v^+) eta(v) m(e) - sum_(e in E_v^-) eta(v) m(e))
  $

  which tracks with the intuitive understanding of the divergence.
]

#lemma[
  The discrete derivative and the adjoint are linear operators.
]

== The Laplacian operator

#definition[Laplacian operator][
  Given a weighted pseudograph $(G, m)$, we define the _laplacion operator_ $Delta: ell_2(V, e) -> ell_2(V, e)$ as

  $
    Delta = dif^* d
  $
]

#theorem[Laplacian operator properties][
  + *$Delta$ is self-adjoint*:
    $Delta = Delta^*$ since
    $
      Delta^* & = (dif^* dif)^* \
              & = dif^* (dif^*)^* \
              & = dif^* dif \
              & = Delta \
    $
  + *$Delta$ is positive semi-definite*:
    $innerproduct(Delta phi, phi) >= 0$ because
    $
      innerproduct(Delta phi, phi) & = innerproduct(dif^* dif phi, phi) \
                                   & = innerproduct(dif phi, dif phi)   & >= 0
    $

  + *$Delta$ is bounded by $2rho_oo = 2 sup_(v in V) rel_m (v)$* #faint[sometimes its $delta(v)$ instead of $rel_m (v)$] as in that $abs(innerproduct(Delta phi, phi)) <= 2 rho_oo$.
    This also implies that any eigenvalue of $Delta$ is at most $2 rho_oo$. Note that $phi$ is an eigenfunction of $Delta$ with eigenvalue $lambda$ iff $Delta phi = lambda phi$.
]

#proof[of 3][
  First, we have to remember Cauchy-Young: $(a - b)^2 <= 2(a^2 + b^2)$. Then, for any arbitrary $phi$, we have

  $
    abs(innerproduct(Delta phi, phi)) & = abs(innerproduct(dif phi, dif phi)) \
    & = norm(dif phi)^2 \
    & = sum_(e in E) abs(dif phi(e))^2 m(e) \
    & = sum_(e in E) abs(phi(endpoint e) - phi(origin e))^2 m(e) \
    & <= 2sum_(e in E) (abs(phi(endpoint e))^2 m(e) + abs(phi(origin e))^2 m(e)) \
    & = 2 sum_(v in V) abs(phi(v))^2 sum_(e in E) m(e) \
    & = 2 sum_(v in V) abs(phi(v))^2 rel_m (v) m(v) \
    & <= 2 sup_(v in V) rel_m (v) sum_(v in V) abs(phi(v))^2 m(v) \
    & = 2 sup_(v in V) delta (v) sum_(v in V) abs(phi(v))^2 m(v) \
    & = 2 rho_oo norm(phi)^2.
  $

  The eigenvalue is also bounded because

  $
    innerproduct(Delta phi, phi) & = innerproduct(lambda phi, phi) \
                                 & = lambda innerproduct(phi, phi) \
                                 & = lambda norm(phi)^2 \
                       => lambda & in [0, 2 rho_oo]
  $
]

#theorem[Laplacian expression][
  The laplacian of a graph $G$ is given explicitly by

  $
    Delta phi(v) & = 1/m(v) sum_(e in E_v) (phi(v) - phi(v_e)) m(e) \
                 & = rel_m (v) phi(v) - 1/m(v) sum_(e in E_v) phi(v_e) m(e)
  $

  where $v_e$ is the vertex opposite to $v$ on the edge $e$.
]

Note that while the derivative and the adjoint depend on the orientation of the edges, the Laplacian does not.

We can calculate the Laplacian of a graph for the combinatorial weight, where we have that $rel_m (v) = delta(v)$, so the Laplacian of $G^"comb"$ is:

$
  Delta^"comb" phi (v) = delta(v) phi(v) - sum_(e in E_v) phi(v_e)
$

If, additionally, we are in a finite graph where $abs(V) < oo$, we have that any $phi in ell_2(V, m)$ is ismorphic to $ell_2(V)$ which is in turn isomorphic to $CC^n$, since you can just write $phi$ as a vector of evaluated values at each vertex. That is,

$
  phi tilde mat(phi(v_1); phi(v_2); dots.v; phi(v_n)) in CC^n
$

Then,

$
  Delta phi & = Delta mat(phi(v_1); phi(v_2); dots.v; phi(v_n)) \
  & = mat(delta(v_1) phi(v_1) - sum_(e in E_v_1) phi(v_1); delta(v_2) phi(v_2) - sum_(e in E_v_2) phi(v_2); dots.v; delta(v_n) phi(v_n) - sum_(e in E_v_n) phi(v_n)) \
  & = mat(delta(v_1) phi(v_1); delta(v_2) phi(v_2); dots.v; delta(v_n) phi(v_n)) - mat(sum_(e in E_v_1) phi(v_1); sum_(e in E_v_2) phi(v_2); dots.v; sum_(e in E_v_n) phi(v_n)) \
  & = underbrace(mat(delta(v_1); , delta(v_2); , , dots.down; , , , delta(v_n)), D^G) mat(phi(v_1); phi(v_2); dots.v; phi(v_n)) - A^G mat(phi(v_1); phi(v_2); dots.v; phi(v_n)) \
  & = D^G phi - A^G phi
$

so we can say that

$
  Delta^"comb" = D^G - A^G
$

where $D^G$ is the diagonal degree matrix, and $A^G$ is just the adjacency matrix. Note that, since $A^G$ and $D^G$ are symmetric, $Delta^"comb"$ is symmetric too.

Now, for the standard weight, we have that $rel_m (v) = 1$, so $Delta^"std" phi(v) = phi(v) - 1/delta(v) sum_(e in E_v) phi(v_e)$. This eventually leads to the following Laplacian matrix:

$
  Delta^"std" = I - (D^G)^(-1/2) A^G (D^G)^(-1/2)
$

#theorem[Laplacian matrices][
  For a finite graph $G = (V, E)$ where $abs(V) < oo$, the laplacians can be expressed as matrices and are given explicitly for the intrinsic weights by


  $
    Delta^"comb" & = D^G - A^G \
     Delta^"std" & = I - (D^G)^(-1/2) A^G (D^G)^(-1/2)
  $
]

#lemma[Congruency of $Delta^"comb"$ and $Delta^"std"$][
  $
    Delta^"comb" & = D^G - A^G \
                 & = (D^G)^(1/2) [I - (D^G)^(-1/2) A^G (D_G)^(-1/2)] (D^G)^(1/2) \
                 & = (D^G)^(1/2) Delta^"std" (D^G)^(1/2) \
  $

  So you can transform between the combinatorial and standard laplacian for finite graphs using similarity transformations.
]

Note: Since $Delta^"comb"$ and $Delta^"std"$ are similar, Sylvester's inertia theorem implies that the number of positive, zero and negative eigenvalues is the same.


#show raw.where(lang: "graph"): enable-graph-in-raw(tree-render)
#show raw.where(lang: "graph"): set text(font: "IosevkaTerm NF")
#example[Laplacians of line graph][
  For $G = P_n$: ```graph   1-2; 2-3; 3-4; ```

  the laplacians are $ Delta^"comb" = D^G - A^G & = mat(1; , 2; , , 2; , , , 2; , , , , dots.down; , , , , , 2; , , , , , , 1) - mat(0, 1; 1, 0, 1; , 1, 0, dots.down; , , 1, dots.down, 1; , , , dots.down, 0, 1; , , , , 1, 0) \
  & = mat(1, -1; -1, 2, -1; , -1, 2, dots.down; , , -1, dots.down, -1; , , , dots.down, 2, -1; , , , , -1, 1) $

  and

  $
    Delta^"std" &= I - (D^G)^(-1/2) A^G (D^G)^(-1/2) \ & =
    mat(1; , 1; , , 1; , , , dots.down; , , , , 1; , , , , , 1) \ &-
    mat(1; , 1 \/ sqrt(2); , , 1 \/ sqrt(2); , , , dots.down; , , , , 1 \/ sqrt(2); , , , , , 1)
    mat(0, 1; 1, 0, 1; , 1, 0, dots.down; , , 1, dots.down, 1; , , , dots.down, 0, 1; , , , , 1, 0)
    mat(1; , 1 \/ sqrt(2); , , 1 \/ sqrt(2); , , , dots.down; , , , , 1 \/ sqrt(2); , , , , , 1) \
    & =
    mat(1, -1\/sqrt(2); -1\/sqrt(2), 1, -1\/sqrt(2); , -1\/sqrt(2), 1, dots.down; , , -1\/sqrt(2), dots.down, -1\/sqrt(2); , , , dots.down, 1, -1\/sqrt(2); , , , , -1\/sqrt(2), 1)
  $
]

#lemma[

  In general, for some graph $G = (V, E)$, we have that

  $
    Delta_(j k)^"comb" = cases(
      & abs(tilde(E)_v_j) quad & "if" j = k,
      - & abs(E(v_j, v_k)) quad & "if" j != k
    )
  $

  $
    Delta_(j k)^"std" = cases(
      & abs(tilde(E)_v_j)/abs(E_v_j) quad & "if" j = k,
      - & abs(E(v_j, v_k))/sqrt(abs(E_v_j) abs(E_v_k)) quad & "if" j != k
    )
  $

  where $abs(tilde(E)_v_j)$ is the degree of $v_j$ without counting loops.
]

== Connectedness and spectrum of the Laplacian

Recall that $0$ is always an eigenvalue of the Laplacian (even for infinite graphs),  since for $phi = 1$ we have that $(Delta phi)(v) = delta(v) - sum_(e in E_v 1) = delta(v) - delta(v) = 0$.

#theorem[
  Let $G$ be an oriented finite graph. The number of connected of components of $G$ is equal to the multiplicity of the $0$ eigenvalue of $Delta^"comb"$.

  In particular, if $0$ is a simple eigenvalue, $G$ is connected.
]

This theorem holds for infinite graphs, but the proof is much more technical. It also holds for the Laplacian using any arbitrary weight (for instance, by Sylvester and the similarity to $Delta^"std"$, it clearly holds for $Delta^"std"$ too).

#proof[
  + $==>$ #todo[Exercise for me]
  + $<==$ #todo[Exercise for me]
]

When a graph is not connected, the eigenvectors $phi$ of $0$ are "piecewise constant", in the sense that for each vertex of each connected component $G_k$ they have a constant value $alpha_k$.

In principle, you cannot just read out whether a matrix is connected from the adjecency matrix directly. However, this does happen for one particular kind of graph...

#definition[$r$-regular graph][
  A weighted graph $(G, m)$ is _$r$-regular_ iff $delta(v) = r$ for all $v in V$.
]

#theorem[
  An $r$-regular graph $(G, m)$ has always eigenvalue $r$.

  Moreover, the number of connected components of $G$ equals the multiplicity of the eigenvalue $r$. In particular, $G$ is connected iff $r$ is a simple eigenvalue.
]

#proof[
  #todo[Exercise for me.]
]

== Bipartiteness and the adjecency matrix

We first need some prerequisites.

#definition[Irreducible matrix][
  A matrix $A$ is _irreducible_ if $exists k in NN$ such that $(A^k)_(i j) != 0$ for all entries of $A$.
]

#lemma[
  If a graph $G$ is connected, then its adjecency matrix $A^G$ is irreducible, since for every $k in NN$ and every $i, j$ the entry $A^k_(i j)$ is the number of paths of length $k$ connecting $v_i$ to $v_j$. In a connected graph, any two vertices have a minimum length path that connects them, so for $k$ equal common multiple of all the min lengths between two vertices the matrix will have no null entries.
]

#theorem[Perron-Frobenius][
  Given an irreducible entrywise nonnegative matrix $A in RR^(n times n)$, the eigenvalue with largest absolute value $lambda$ of $A$ is positive, simple and has an entrywise positive eigenvector.
]

#theorem[Min-max principle][
  Given a Hermitian matrix $M in CC^(n times n)$ which has real eigenvalues $sigma(M) = { lambda_1, lambda_2, ..., lambda_n }$ (labelled in increasing order),

  $
    lambda_k & = min_(E_k in xi_k) max_(x in E_k) innerproduct(M x, x) / innerproduct(x, x) \
    & = max_(E_k in xi_k) max_(x in E_k \ norm(x) = 1) innerproduct(M x, x)
  $

  where $xi_k$ is the set of all subspaces of $CC^n$ of dimension $k$:

  $
    xi_k = { E_k subset CC^n : dim E_k = k }.
  $

  In particular,

  $
    lambda_1 = min_(x in CC^n \ norm(x) = 1) innerproduct(M x, x) \
    lambda_n = max_(x in CC^n \ norm(x) = 1) innerproduct(M x, x) \
  $

  #faint[Note: $innerproduct(M x, x) / innerproduct(x, x)$ is called the _Rayleight quotient_.]
]

The above theorem provides a complete description of every eigenvalue!

#proof[
  #todo[Exercise for me]
]

And now we can get to the main theorem:

#theorem[
  Given a directed pseudograph $G = (V, E)$ with adjecency matrix $A^G = A$,
  + If $G$ is connected, given the largest eigenvalue $m_1$ of $A$, if $-mu_1$ is an eigenvalue of $A$, then $G$ is bipartite.
  + For a finite grapg $abs(V) < oo$ and the spectrum $sigma(A)$, the following are equivalent:
    + $G$ is bipartite
    + $sigma(A)$ is symmetric (i.e., $mu in sigma(A) <=> -mu in sigma(A)$)
    + $mu_k = -mu_(n - k + 1)$ for $k = 1, ..., n$ (in particular, $0 in sigma(A)$ if $n$ is odd).
]

#proof[
  #todo[Exercise for me]
]
