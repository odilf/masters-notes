#import "../format.typ": *

#set text(font: "New Computer Modern")

#set align(horizon)
#show: exercises[Exercises for Discrete Applied Mathematics]
#show outline.entry: it => link(
  it.element.location(),
  it.indented(strong(it.prefix()), it.inner()),
)
#outline()
// #pagebreak(weak: true)

#set align(top)

#set heading(numbering: none)
#show heading.where(level: 1): set heading(
  numbering: i => [Exercise Sheet #i.],
  supplement: "Exercise sheet",
)
#show heading.where(level: 2): set heading(
  numbering: "1.",
  // numbering: i => [Exercise Sheet #i.],
  supplement: "Problem",
)

#show heading.where(level: 1): it => pagebreak(weak: true) + it
#show heading.where(level: 2): it => v(2cm, weak: true) + it

#import "@preview/h-graph:0.1.0": (
  enable-graph-in-raw, polar-render, tree-render,
);
#show raw.where(lang: "graph"): enable-graph-in-raw(tree-render)
#show raw.where(lang: "graph"): set text(font: "IosevkaTerm NF")


#counter(heading).update(_ => 4)

= Spectrum of the adjacency matrix

== Connected components

Let $G = (V, E, partial)$ be a (finite) multidigraph. Suppose that $G$ has $r in NN$ connected components $G_i$.

#set enum(numbering: i => [*#numbering("(i)", i)*])
+ *Show that $A^G tilde.equiv A^(G_1) plus.o A^(G_2) plus.o ... plus.0 A^(G_r)$ and $sigma(A^G) = union_(i=1)^r sigma(A^(G_r))$, and similarly for $Delta$.*

  We can write $A^G$ as a block-diagonal matrix containing all $A^(G_i)$ and $Delta$ as a block-diagonal of $Delta_i$s.

+ *Show that $lambda_1(Delta) = 0$  is an eigenvalue of $Delta$.*

  If we take $phi = bb(1) = mat(1; 1; dots.v; 1)$, then $ Delta phi (v) = 1/m(v) sum_(e in E_v) (phi(v) - phi(v_e)) m(e) = 1/m(v) sum_(e in E_v) 0 = 0 $

  Therefore, $phi = bb(1)$ is an eigenvector of $phi$ with associated eigenvalue $0$.

== Counting number of walks

Let G = (V, E) be a simple graph with vertices { v_1, ..., v_n } and denote by $A$ the adjecency matrix.

+ *Show that the matrix elements $A_(i j)$ and $(A^2)_(i j)$ count the number of walks of length $1$ and $2$, respectively, that start on $v_i$ and end on $v_j$. How do you interpret in this context that $A$ is symmetric?*

  The walks of length $1$ is essentially the definition of the adjecency matrix. It is $1$ if the two vertices are connected and $2$ if it's the same vertex and it has a self-loop. Now, if we evaluate where $(A^2)_(i j)$ comes from when doing matrix multiplication, we can see that

  $
    (A^2)_(i j) = A_(i,:) dot A_(:,j) = sum_(k=0)^n A_(i k) A_(k j)
  $

  that is, the $i$th row of $A$ dotted with the $j$th column of $A$. This operation is "scanning" through all vertices $v_k$ and adding one if both $i$ is adjacent to $k$ and $j$ is also adjacent to $k$, which forms a path of length $2$ from $i$ to $j$.

  The fact that it is symmetric means that the graph is not oriented, since a walk from $i$ to $j$ is the same as a walk from $j$ to $i$.

+ *Show by induction that the matrix element $(A^k)_(i j)$ counts the number of walks of length $k$ that start in $v_i$ and end in $v_j$ (denoted by $v_i ->_k v_j$)*

  We have the base case. Now, assuming that $(A^(k-1)_(i j))$ counts $v_i ->_(k-1) v_j$, multiplying by $A$ gives:

  $
    (A^k)_(i j) = sum_(q = 0)^n (A^(k-1))_(i q) A_(q j)
  $

  which, for both to be $1$, there needs to be a path of length $k-1$ from $i$ to $q$, and $q$ needs to be adjacent to $j$, which forms a path of length $k$ from $i$ to $j$. Therefore, $(A^k)_(i j)$ holds the number of walks of length $k$ from $i$ to $j$.

== Spectra of standard examples

Compute the spectra of the adjecency and Laplace matrices (with combinatorial weights) associated to the following graphs:

+ *The path graph, $P_n$*
+ *The cyclic graph, $C_n$*
+ *The complete graph, $K_n$*

#todo[bruh]

== Adjecency matrix and isospectral graphs

Let $G = (V, E, partial)$ be an oriented graph with combinatorial weights, and let $A$ be the adjecency matrix.

+ *Show that the following two pairs of graphs are isospectral (i.e., have the same spectrum), but do have different number of connected components.*
  + *$G_1 = K_(1, 4)$ and $G_2 = K_1 union.sq C_4$*

    Clearly $G_1$ is connected while $G_2$ has two connected components. First, we write out both adjecency matrices:

    $
      A_1 = mat(
        0, 1, 1, 1, 1;
        1, 0, 0, 0, 0;
        1, 0, 0, 0, 0;
        1, 0, 0, 0, 0;
        1, 0, 0, 0, 0;
      )
    $

    and

    #todo[I did all the cyclic graphs wrong...]

    $
      cancel(
        A_2 = mat(
          0, 0, 0, 0, 0;
          0, 0, 1, 0, 0;
          0, 0, 0, 1, 0;
          0, 0, 0, 0, 1;
          0, 1, 0, 0, 0;
        )
      )
    $

    Now, if $phi$ is an eigenvector of $A_1$, we have that $A_1 phi = mu phi$, so

    $
      mu mat(phi_1; phi_2; phi_3; phi_4; phi_5) = mat(phi_2 + phi_3 + phi_4 + phi_5; phi_1; phi_1; phi_1; phi_1)
    $

    which we can rearange as $mu phi_1 = 1/mu 4 phi_1 => mu^2 = 4 => mu = plus.minus 2$ if $phi_1 != 0$. If $phi_1 = 0$, then $phi = bb(0)$ identically which is not a valid eigenvector.

    And for $A_2$, $A_2 phi = mu phi$ implies that

    $
      mu mat(phi_1; phi_2; phi_3; phi_4; phi_5) = mat(0; phi_3; phi_4; phi_5; phi_2)
    $

    which in turn implies that either $mu = 0$, or $phi_1 = 0$ and then $mu phi_j = phi_(j + 1)$ (for $phi_6 := phi_2$), so $phi_2 = mu^4 phi_2$, which can only hold if $mu = plus.minus 1$.

    Therefore, the spectrum of both matrices is proportional, $sigma(A_1) = {-2, 0, 2}$, $sigma(A_2) = {-1, 0, 1}$ and $sigma(A_1) = 2sigma(A_2)$.

  + *$G_1$ is the graph of seven vertices, one central of degree $3$ connected with three vertices of degree 2 connected each with a vertex of degree $1$; and $G_2 = K_1 union.sq C_6$*

    Visually,
    ```graph
    1 - 2;
    1 - 3;
    1 - 4;

    2 - 5;
    3 - 6;
    4 - 7;
    ```

    The adjecency matrices are

    $
      A_1 = mat(
        0, 1, 1, 1, 0, 0, 0;
        1, 0, 0, 0, 1, 0, 0;
        1, 0, 0, 0, 0, 1, 0;
        1, 0, 0, 0, 0, 0, 1;
        0, 1, 0, 0, 0, 0, 0;
        0, 0, 1, 0, 0, 0, 0;
        0, 0, 0, 1, 0, 0, 0;
      )
    $

    and

    #todo[This cyclic graph is also wrong]

    $
      cancel(
        A_2 = mat(
          0, 0, 0, 0, 0, 0, 0;
          0, 0, 1, 0, 0, 0, 0;
          0, 0, 0, 1, 0, 0, 0;
          0, 0, 0, 0, 1, 0, 0;
          0, 0, 0, 0, 0, 1, 0;
          0, 0, 0, 0, 0, 0, 1;
          0, 1, 0, 0, 0, 0, 0;
        )
      )
    $


    where for an eigenvalue $A_1 phi = mu phi$ we get the constraints

    $
      mu mat(phi_1; phi_2; phi_3; phi_4; phi_5; phi_6; phi_7) = mat(phi_2 + phi_3 + phi_4; phi_1 + phi_5; phi_1 + phi_6; phi_1 + phi_7; phi_2; phi_3; phi_4)
    $

    which reduce to

    $
            mu phi_1 & = sum_(j = 1, 2, 3) phi_j \
      (mu - 1) phi_j & = phi_1 quad "for" j=2,3,4 \
         => mu phi_1 & = 3 1/(mu - 1) phi_1 \
         mu (mu - 1) & = 3 \
       mu^2 - mu - 3 & = 0 \
                  mu & = (1 plus.minus sqrt(13))/2 \
    $


    Then, for $A_2$, we get

    $
      mu mat(phi_1; phi_2; phi_3; phi_4; phi_5; phi_6; phi_7) = mat(0;)
    $



+ *Show that for an $r$-regular weighted graph $G$, one has that $r$ is an eigenvalue of $A$. Show, in addition, that $G$ is connected iff $r$ is a simple eigenvalue of $A$. Does this result contradict the previous part?*

  Proof for first part of the question is in notes, but the gist is that $Delta = A - r I$, so the theorems that hold for the $0$ eigenvalue in $Delta$ hold for the $r$ eigenvalue in $A$ (if $r$-regular).

  This does not contradict anything since none of the graphs above are regular. In part 1, $G_1$ has degrees 1 and 4 and $G_2$ has degrees 0 and 2; and in part 2, $G_1$ has degrees 1, 2 and 4, and $G_2$ has degrees 0 and 2.


