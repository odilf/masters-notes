#import "../format.typ": *
#import "@preview/physica:0.9.5": evaluated

#show: exercises[]

#exercise-counter.update(1)

#exercise[
  Prove that the following spaces are normed linear spaces $(X, norm(dot))$

  ii) $X = C[a, b]$ and, for $p = 1$ and $p = 2$

  $ ||f|| = ||f||_p := (integral_a^b |f(x)|^p dx)^(1/p) $
][

]

#exercise-counter.update(6)

#exercise[
  Let $f : [−1, 1] -> RR$ be the function defined by

  $
    f(x) = cases(
      0\, & x = 0,
      e^(-1 \/ x^2)\, quad & x != 0
    )
  $

  Prove that

  $
    lim_(k -> oo) norm(f^((k)))_oo := lim_(k -> oo) sup_(x in [-1, 1]) |f^((k)) (x)| = oo
  $
][
  We have seen that if every deriviative of a function is bounded by some constant in a region then the (full) Taylor expansion of that function is equivalent to the original expression.

  Suppose that such a constant exists, so that $sup_(x in [-1,1]) |f^((k)) (x)|= C$. If so, we would be able to do a Taylor expansion of the function. However, we have seen in class that the (complete) Taylor expansion $T_f$ of $f$ is just $T_f (x) = 0$.

  Clearly $T_f != f$. For instance $f (1/2) = e^(-4) != 0 = T_f (1/2)$. Therefore, a bound $C$ for all the derivatives of $f$ cannot exist. Hence, the supremum of $f^((k))$ will increase arbitrarily so the limit as $k -> oo$ is $oo$.
]

#exercise[
  Let X be the space of all polynomials with real coefficients, and consider the
  inner product

  $
    innerproduct(f, g) = integral_(-infinity)^infinity f(x) g(x) e^(-x^2) dif x
  $

  Show that the Hermite polynomials, defined as, $H_n (x) = (−1)^n e^(x^2) (dif^n)/(dif x)^n e^(-x^2)$ form an orthogonal set with respect to this inner product.

  Show that $norm(H_n)^2 = innerproduct(H_n, H_n) = 2^n n! sqrt(pi)$.
][
  Derivatives of $e^(-x^2)$:

  #let ex2 = $e^(-x^2)$
  0. $ex2$
  1. $-2x ex2$
  2. $(-2 + 4x^2) ex2$
  3. $(8x + (-2 + 4x^2) dot (-2x)) ex2$

  The recursive relation seems to be that you always have an expression of $p_n (x) ex2$ where $p_n (x)$ is a polynomial. The relation is $(p_n (x) ex2)' = (p'_n (x) -2x p_n (x)) ex2$ so the recursive relation between polynomials is $p_(n+1) (x) = p'_n (x) -2x p_n (x)$. This makes it easier to enumerate:

  0. $1$
  1. $-2x$
  2. $-2 + 4x^2$
  3. $4x + 4x - 8x^3 = 8x - 8x^3$
  4. $4 + 4 - 24x^2 - 8x^2 - 8x^2 + 16x^4 = 8 -40x^2 + 16x^4$
  5. $-48x - 16x - 16x + 64x^3 - 8x -8x + 48x^3 + 16x^3 + 16x^3 - 32x^5 = -96x + 144x^3 - 32x^5$

  For each $p_n$, we get one side with terms reduced via $p'_n$ and another side with terms augmented via $2x p_n$.

  What is important here is to notice that the structure of the numbers is

  $ p_n (x) & = sum_(k in S_n) c_k x^k $

  Where $S_n = { k | k < n, k = 2 space (mod 2)}$. That is, it has only powers of the parity of $n$.

  Or, to be more explicit:

  $
    p_n (x) = cases(
      sum_(k=0)^(n/2) c_k x^(2k) & " if" n "is even",
      sum_(k=0)^((n-1)/2) c_k x^(2k + 1) & " if" n "is odd"
    )
  $

  So
  $
    p_(n+1) (x) = p'_n (x) -2x p_n(x)
    & = (sum_(k in S_n) c_k x^k)' - 2x sum_(k in S_n) c_k x^k \
    & = sum_(k in S_n) c_k k x^(k - 1) - 2 sum_(k in S_n) c_k x^(k + 1) \
  $


  To prove the original statements, we will show that $innerproduct(H_n, H_m) = 0$ if $m > n$, which by symmetry implies that $innerproduct(H_n, H_m) = 0$ if $n != m$.

  We start with the base case $innerproduct(H_n, H_(n + 1))$. This is:

  $
    innerproduct(H_n, H_(n+1)) & = integral_(-oo)^oo p_n (x) p_(n+1)(x) ex2 dx \
    & = integral_(-oo)^oo p_n (x) (p'_n (x) - 2x p_n (x)) ex2 dx \
    & = integral_(-oo)^oo p_n (x) p'_n (x) ex2 dx + integral_(-oo)^oo - 2x (p_n (x))^2 ex2 dx \
    & = integral_(-oo)^oo sum_(k,l in S_n) c_k c_l x^(k + l - 1) ex2 dx + integral_(-oo)^oo - 2x (p_n (x))^2 ex2 dx \
  $

  // ---

  // Generally, we get something of the form $ex2 p_n (x)$ where $p$ is a polynomial, so $H_n (x) = e^(x^2) ex2 p_n (x) = p_n (x)$.

  // Leading term of $dif^n/(dif x)^x ex2$ is $(-2 x)^n ex2$. $n^"th"$ derivative has polynomial where the parity of the powers match $n$.

  // Clearly, if $x -> f(x) g(x)$ is odd, then $innerproduct(f, g) = 0$. However, here we expect that something like $innerproduct(H_0, H_2) = 0$ is also $0$, even though it is an even function. Let's compute it:

  // $
  //   innerproduct(H_0, H_2) & = integral_(-infinity)^infinity (1) (-2 + 4x^2) ex2 dif x \
  //   & = integral_(-infinity)^infinity -2 ex2 dif x + integral_(-infinity)^infinity 4x^2 ex2 dif x \
  //   & = -2 sqrt(pi) + 2(-2x^2 ex2|_(-infinity)^infinity - integral_(-infinity)^infinity -2 ex2) \
  //   & = -2 sqrt(pi) + 2(sqrt(pi)) \
  //   & = 0 \
  // $

  // This is kind of fucked up.

  // We can do this another way. We can prove that $innerproduct(H_n, H_m) = 0$ if $m > n$. This gives us for free that $innerproduct(H_m, H_n) = 0$ because of symmetry.

  // So,

  // $ innerproduct(H_n, H_m) = integral_(-infinity)^(infinity) p_n (x) p_m (x) ex2 dx $.
]

#exercise-counter.update(8)
#exercise[
  Let $f(x) = e^x$. Give explicitly the polynomial of degree $2$ that best approximates $f$ with respect to the norm

  $ norm(g) = (integral_(-infinity)^infinity |g(x)|^2 e^(-x^2) dif x)^(1/2) $

  In other words: for which polynomial p of degree 2 is $norm(f − p)$ minimal? What is the minimal error?
][
  To solve this exercise we:
  + Prove that the norm comes from an inner product
  + Make an orthonormal basis for polynomials of degree 2 given that inner product
  + Find the coefficients for the basis that makes $norm(f - p)$ minimal.
  + Compute $norm(f - p)$

  == Step 1: norm as inner product

  The norm can be constructed from an inner product $innerproduct(g, g)^(1/2)$, where

  $ innerproduct(f, g) = integral_(-oo)^oo f(x) g(x) e^(-x^2) dx $

  We prove that this satisfies all properties of inner products:

  *i)* $innerproduct(f, f) > 0$ if $f != 0$ and $innerproduct(f, f) = 0$ if $f = 0$.

  Let $f(x) = a + b x + c x^2$. If $f = 0$ then $innerproduct(0, 0) = integral_(-oo)^oo 0 dx = 0$. If $f != 0$ then one of the coefficients of $f(x) = a + b x + c x^2$ will be non-zero. $e^(-x^2)$ is strictly positive and $(f(x))^2$ is also for any non-zero coefficient of either $a$, $b$ or $c$ #todo[Does this need more explanation?]

  *ii)* $innerproduct(f, g) = innerproduct(g, f)$

  $
    innerproduct(f, g) = integral_(-oo)^oo |f(x)g(x)| e^(-x^2)dx = integral_(-oo)^oo |g(x)f(x)| e^(-x^2) dx = innerproduct(g, f)
  $

  *iii)* $innerproduct(alpha f + beta g, h) = alpha innerproduct(f, h) + beta innerproduct(g, h)$

  $
    innerproduct(alpha f + beta g, h)
    & = integral_(-oo)^oo (alpha f(x) + beta g(x)) h(x) e^(-x^2) dx \
    & = integral_(-oo)^oo alpha f(x)h(x)e^(-x^2) dx + integral_(-oo)^oo beta g(x)h(x)e^(-x^2) dx \
    & = alpha integral_(-oo)^oo f(x)h(x)e^(-x^2) dx + beta integral_(-oo)^oo g(x)h(x)e^(-x^2) dx \
    & = alpha innerproduct(f, h) + beta innerproduct(g, h)
  $

  Since the norm comes from an inner product, we can find the best approximation given an orthonormal basis $G = { g_0, g_1, g_2 }$ as $sum_(g_k in G) innerproduct(g_k, f) g_k$.

  == Step 2: orthonormal basis

  First, we need to find an orthonormal basis. We start with a linearly independent basis for polynomials of degree $2$ which is ${ 1, x, x^2 }$. Then, we orthogonalize it. $1$ and $x$ and $x$ and $x^2$ are already orthogonal, since:

  #let int = $integral_(-oo)^oo$
  #show math.eq: math.limits

  $
    innerproduct(1, x) = int 1 dot x e^(-x^2) dx = evaluated(-1/2 e^(-x^2))_(-oo)^oo = 0
  $

  $
    innerproduct(x, x^2) = int x^3 e^(-x^2) dx =^"by parts" evaluated(-1/2 x^2 e^(-x^2))_(-oo)^oo + int x e^(-x^2) dx = 0
  $

  However, $1$ and $x^2$ are not orthogonal. That is,

  $
    innerproduct(1, x^2) = int x^2 e^(-x^2) dx =^"by parts" evaluated(-1/2 e^(-x^2))_(-oo)^oo + 1/2 int e^(-x^2) dx =^"well known" sqrt(pi)/2
  $

  Therefore, we have to apply the Gram Schmidt procedure. Before doing it we are going to normilize the other polynomials first to reduce computations. That is:

  $ norm(1) = sqrt(innerproduct(1, 1)) = (int e^(-x^2) dx)^(1/2) = pi^(1/4) $
  $
    norm(x) = sqrt(innerproduct(x, x)) = (int x^2 e^(-x^2) dx)^(1/2) = pi^(1/4)/sqrt(2)
  $

  #faint[we know the integral of $x^2 e^(-x^2)$ from computing $innerproduct(1, x^2)$ before]
  #let g0 = $1/pi^(1/4)$
  #let g1 = $(x sqrt(2))/(pi^(1/4))$

  Therefore our two first elements of the basis are $g_0 = g0$ and $g_1 = g1$.

  We can now apply Gram Schmidt to $x^2$. from linearity we have that $innerproduct(g0, x^2) = sqrt(pi) / 2 div pi^(1/4) = pi^(1/4)/2$ and that, still, $innerproduct(g1, x^2) = 0$, so a third orthogonal element is

  $ x^2 - pi^(1/4)/2 g_0 = x^2 - 1/2 $

  We still have to normalize this element to get the final basis element, so we have:

  $
    innerproduct(x^2 - 1/2, x^2 - 1/2)
    &= int (x^2 - 1/2)^2 e^(-x^2) dx \
    &= int (x^4 - x^2 + 1/4) e^(-x^2) dx \
    &= int x^4e^(-x^2) dx - int x^2 e^(-x^2) dx + 1/4 int e^(-x^2) dx \
    &= 3/2 int x^2 e^(-x^2) dx - sqrt(pi)/2 + sqrt(pi)/4 \
    &= (3 sqrt(pi))/4 - sqrt(pi)/4 \
    &= sqrt(pi)/2 \
  $

  So $norm(x^2 - 1/2) = sqrt(innerproduct(x^2 - 1/2, x^2 - 1/2)) = pi^(1/4)/sqrt(2)$, with which we can get our final basis element:

  #let g2 = $(x^2 - 1/2) sqrt(2)/pi^(1/4)$
  $ g_2 = g2 $

  The basis is, then:

  $ G = { g_0, g_1, g_2 } = { g0, g1, g2 } $

  == Step 3: Coefficients

  Now we compute the coefficients. First, let's calculate a common integral by completing the square. Namely, the integral of $e^(-x^2 + x)$, which is:

  $
    int e^(x - x^2) dx & = int e^(-(x-1/2)^2 + 1/4) dx \
                       & = e^(1/4) int e^(-(x-1/2)^2) dx \
                       & = e^(1/4) pi^(1/2) \
  $

  Now let's compute the coefficients. Remember that $c_k = innerproduct(f, g_k)$, so

  $
    c_0 = innerproduct(f, g_0) = innerproduct(e^x, g0) & = int e^x g0 e^(-x^2) dx \
                                                       & = g0 int e^(x - x^2) dx \
                                                       & = g0 e^(1/4) pi^(1/2) \
                                                       & = (pi e)^(1/4) \
  $

  $
    c_1 = innerproduct(f, g_1) & = innerproduct(e^x, g1) \
    &= int g1 e^(x - x^2) dx \
    &= sqrt(2) / pi^(1/4) int x e^(x - x^2) dx \
    &=^"by parts" sqrt(2) / pi^(1/4) ( cancel(evaluated(-1/2 e^(x-x^2))_(-oo)^oo) + 1/2 int e^(x - x^2) dx ) \
    &= sqrt(2) / pi^(1/4) 1/2 e^(1/4) pi^(1/2) \
    &= sqrt(2)/2 (pi e)^(1/4) \
  $

  And

  $
    c_2 = innerproduct(f, g_2) &= innerproduct(e^x, g2) \
    &= int g2 e^(x - x^2) dx \
    &= sqrt(2)/pi^(1/4) (int x^2 e^(x - x^2) dx - 1/2 int e^(x - x^2) dx ) \
    &= sqrt(2)/pi^(1/4) (cancel(evaluated(-1/2 x e^(x-x^2))_(-oo)^oo) - int (e^x + x e^x) (-1/2 e^(-x^2)) dx - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) (1/2 int (e^x + x e^x) e^(-x^2) dx - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) (1/2 (int e^(x-x^2) dx + int x e^(x-x^2) dx) - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) (1/2 (e^(1/4) pi^(1/2) + 1/2 e^(1/4) pi^(1/2)) - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) (1/2 (3/2 e^(1/4) pi^(1/2)) - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) (3/4 e^(1/4) pi^(1/2) - (e^(1/4) pi^(1/2))/2 ) \
    &= sqrt(2)/pi^(1/4) 1/4 e^(1/4) pi^(1/2) \
    &= sqrt(2)/4 (pi e)^(1/4) \
  $

  So, the best polynomial approximation to $f$ with the given norm is

  $
    p(x) = (pi e)^(1/4) (1 + sqrt(2)/2 x + sqrt(2)/4 (x^2 - 1/2))
    \ = (pi e)^(1/4) (1 - sqrt(2)/8 + sqrt(2)/2 x + sqrt(2)/4 x^2)
    \ = sqrt(2) (pi e)^(1/4) (7/8 + x/2 + x^2/4)
  $

  == Step 4: Computing distance

  The distance $norm(e^x - p(x))$ is:

  $ norm(e^x - sqrt(2) (pi e)^(1/4)(7/8 + x/2 + x^2/4)) $

  This would be a monstrosity to compute, but we can instead use the Pythagorean theorem for inner product spaces. We can manipulate a bit the expression $norm(e^x - p)^2 = innerproduct(e^x - p, e^x - p)$:
  $
    innerproduct(e^x - p, e^x - p) & = 
    innerproduct(e^x, e^x - p) - innerproduct(p, e^x - p) \ & = 
    innerproduct(e^x, e^x) - innerproduct(e^x, p) - innerproduct(p, e^x)  + innerproduct(p, p) \ & =
    innerproduct(e^x, e^x) - 2 innerproduct(e^x, p) + innerproduct(p, p)
  $

  Here we have three distinct terms. The first is simple:

  $ innerproduct(e^x, e^x) = 1 $

  For the second, we can break it down into the basis vectors and use orthogonality:

  $ innerproduct(e^x, p) & = innerproduct(e^x, c_0 g_0 + c_1 g_1 + c_2 g_2)
  \ & = c_0 innerproduct(e^x, g_0) + c_1 innerproduct(e^x, g_1) + c_2 innerproduct(e^x, g_2)
   $

  However, the inner products that multiply the coefficients are just the definition of the coefficients themselves! So:

  $ innerproduct(e^x, p) = c_0^2 + c_1^2 + c_2^2 $

  Finally, we need the term $innerproduct(p, p)$ which is also the sum of the coefficients, but this time just by simple Pythagorean theorem.

  #faint[Another way to put it is that $e^x - p$ and $p$ are perpendicular since that is what makes a best approximation, so $norm(p + e^x - p)^2 = norm(p)^2 + norm(e^x - p)^2 => norm(e^x - p)^2 = norm(e^x)^2 - norm(p)^2$. This is simpler but I didn't invoke it because we haven't explicitly stated in the course that a best approximation is the orthogonal projection, and the derivation for that is basically what I have written above.]

  Therefore, the inner product is:

  $ innerproduct(e^x - p, e^x - p) & = 1 - c_0^2 + c_1^2 + c_2^2
  \ & = 1 - (pi e)^(1/4) (1 + sqrt(2)/2 + sqrt(2) / 4)
  \ & = 1 - (pi e)^(1/4) (1 + 3sqrt(2)/4)
   $

  This value is the norm squared, so the distance is:

  $ norm(e^x - p) = sqrt(innerproduct(e^x - p, e^x - p)) = sqrt(1 - (pi e)^(1/4) ( 1 + 3sqrt(2)/4 )) $
]

#exercise[
  Which polynomial $p$ of degree 2 best approximates the function $|x|^(−1/3)$ on $[-1, 1]$ with respect to the norm

  $ norm(f) = (integral_(-oo)^oo |f(x)|^2 dx)^(1/2) $

  What is the minimal error? Can $|x|^(-1/3)$ be approximated by polynomials with respect to the supremum norm to arbitrary precision?
][

]

#exercise[
  Consider the 2π-periodic function $ f: RR -> RR $ defined by

  $ f(x) = pi^2/3 - pi|x| + 1/2x^2, quad x in (-pi, pi] $

  Find a number $n$ and a trigonometric polynomial

  $ S_n (x) = 1/2 a_0 + sum_(k=1)^n (a_k cos(k x) + b_k sin(k x)) $

  with real coefficients such that

  $ norm(f - S_n)_oo := sup_(x in (-pi, pi]) |f(x) - S_n (x)| < 10^(-1) $

  Plot $f$ and $S_n$ to visualize the error.
][

]
