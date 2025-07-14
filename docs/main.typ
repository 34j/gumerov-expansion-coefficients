#import "@preview/algorithmic:0.1.0"
#import "@preview/physica:0.9.2": *
#import "@preview/ctheorems:1.1.3": *
#import "@preview/cetz:0.3.1"
#import "@preview/colorful-boxes:1.4.3": colorbox

#import algorithmic: algorithm
#let thmboxargs = (inset: 0em, padding: (top: 0em, bottom: 0em))
#let definition = thmbox("definition", "Definition", ..thmboxargs)
#let theorem = thmbox("theorem", "Theorem", ..thmboxargs)
#let lemma = thmbox("lemma", "Lemma", ..thmboxargs)
#let corollary = thmbox("corollary", "Corollary", ..thmboxargs)
#let example = thmbox("example", "Example", ..thmboxargs)
#let axiom = thmbox("axiom", "Axiom", ..thmboxargs)
#let remark = thmbox("remark", "Remark", ..thmboxargs)
#let proof = thmproof("proof", "Proof")
#let calculated(..args) = colorbox(..args, title: "Calculated")
#show: thmrules

// functions
#let hk1 = $h^((1))$

// S|R
#let ssc = $(S|S)$
#let src = $(S|R)$
#let rrc = $(R|R)$
#let ty = $op(cal(T))$
#let efc = $(E|F)$
#let sign = $op("sign")$
#let ip(x, y) = $lr(( #x, #y ))$

= Translation

#definition[
  $
    efc^(m' n)_(n' n) &= (2 pi)^(3 / 2) sqrt(2/pi) sum_(n'',m'') (-i)^(n - n' - n'') cases(R &(E = F),S &(E != F))^(m'')_(n'') (t) \
    &quad times integral_(SS^(d-1)) Y^m_n (x) overline(Y_n'^m' (x)) overline(Y_n''^m'' (x)) dd(x)
  $
  $
    efc^(m' m)_(n', ) := efc^(m' m)_(n', abs(m))
  $
  $
    efc^(m' m)_(, n) := efc^(m' m)_(abs(m'), n)
  $
]
#theorem[
  $
    not (abs(m) <= n and abs(m') <= n') ==> efc^(m' m)_(n' n) = 0
  $
] <zero>
#theorem[2.18-19][
  $
    E^m_n (x + t) = sum_(m', n') efc^(m' m)_(n' n) (t) F^m'_n' (x)
  $
]

== Main part

#colorbox(
  title: "Goal",
  color: (stroke: maroon),
)[Calculate $efc^(m',m)_(n',n)$ for $abs(m') <= n' <= N and abs(m) <= n <= N$]

#theorem[4.43, 4.58][
  $
    efc^(m' 0)_(n' 0) (t) = 1_(abs(m') < n) sqrt(4 pi) (-1)^n cases(R &(E = F),S &(E != F))^(-m')_n' (t)
  $
] <init>

#colorbox(
  title: "Calculated",
  color: (stroke: red),
)[
  $efc^(m',m)_(n',n)$ for $abs(m') <= n' <= 2 N and m = n = 0$
]

#theorem[4.30][
  $
    b^m_n underbrace(efc^(m' m+1)_(n' n-1),m = n ==> 0 "(4.31)") - b^(-m-1)_(n+1) efc^(m' m+1)_(n' n+1) = b^(m'-1)_(n'+1) efc^(m'-1 m)_(n' + 1 n) - b^(-m')_(n') efc^(m'-1 m)_(n' - 1 n)
  $
  $
    b^m_n = 1_(abs(m) <= n) underbrace(cases(1 &(m >= 0),-1 &(m < 0)),"Not" sign(m) "!") sqrt(((n-m-1)(n-m))/((2n-1)(2n+1)))
  $
] <b1>

#cetz.canvas({
  import cetz.draw: *
  let l = 4.8
  let k = 4
  ortho(
    x: 18deg,
    y: -30deg,
    z: 0deg,
    {
      line((0, 0, 0), (0, 0, l), mark: (end: ">"))
      line((0, -l, 0), (0, l, 0), mark: (end: ">"))
      line((0, 0, 0), (l, 0, 0), mark: (end: ">"))
      on-xz({
        grid((0, 0), (l, l), step: 1.0, stroke: gray + 0.2pt)
      })
      on-xy({
        line(
          (0, -0.2),
          (0, 0 + 0.2),
          (k, k + 0.2),
          (k, -k - 0.2),
          close: true,
          fill: red.transparentize(50%),
          stroke: none,
        )
        line(
          (0, 0.2),
          (k, k + 0.2),
          (0, k + 0.2),
          close: true,
          fill: gray.transparentize(50%),
          stroke: none,
        )
        line(
          (0, -0.2),
          (k, -k - 0.2),
          (0, -k - 0.2),
          close: true,
          fill: gray.transparentize(50%),
          stroke: none,
        )
        grid((0, -l), (l, l), step: 1.0, stroke: gray + 0.2pt)
        content((k * 0.3, -k * 0.7), [0 (@zero)])
        content((k * 0.7, -k * 0.3), [@init], anchor: "west")
      })
      for i in range(1, k) {
        on-xy(
          {
            line(
              (0, -0.2),
              (0, 0.2),
              (k - i, k - i + 0.2),
              (k - i, -k + i - 0.2),
              close: true,
              fill: blue.transparentize(50%),
              stroke: none,
            )
          },
          z: i,
        )
      }
      on-yz({
        grid((0, -l), (l, l), step: 1.0, stroke: gray + 0.2pt)
      })
      anchor("x2", (1, 2, 2))
      anchor("x3", (2, 1, 1))
      anchor("x4", (0, 1, 1))
      circle("x2", fill: black, radius: 0.1)
      circle("x3", fill: black, radius: 0.1)
      circle("x4", fill: black, radius: 0.1)
      content("x2", [2: $(n',m',n+1)$], anchor: "west")
      content("x3", [3: $(n'+1,m'-1,n)$], anchor: "east")
      content("x4", [4: $(n'-1,m'-1,n)$], anchor: "east")
      content((l, 0, 0), $n'$, anchor: "west")
      content((0, l, 0), $m'$, anchor: "west")
      content((0, 0, l), $m = n$, anchor: "west")
      line("x3", "x2", mark: (end: ">"))
      line("x4", "x2", mark: (end: ">"))
    },
  )
})

#colorbox(
  title: "Calculated",
  color: (
    stroke: blue,
  ),
)[$efc^(m',m)_(n',n)$ for $abs(m') <= n' <= 2 N - n and m = n <= N$
]

#theorem[4.34 (swapped)][
  $
    b^m'_n' efc^(m'+1 m)_(n'-1 n) - b^(-m'-1)_(n'+1) efc^(m'+1 m)_(n'+1 n) = b^(m-1)_(n+1) efc^(m' m-1)_(n' n+1) - b^(-m)_n underbrace(efc^(m' m-1)_(n' n-1), m = -n ==> 0)
  $
  $
    b^m_n = 1_(abs(m) <= n) sign(m) sqrt(((n-m-1)(n-m))/((2n-1)(2n+1)))
  $
] <b2>

#cetz.canvas({
  import cetz.draw: *
  let l = 4.8
  let k = 4
  ortho(
    x: 18deg,
    y: -30deg,
    z: 0deg,
    {
      line((0, 0, 0), (0, 0, l), mark: (end: ">"))
      line((0, -l, 0), (0, l, 0), mark: (end: ">"))
      line((0, 0, 0), (l, 0, 0), mark: (end: ">"))
      on-xz({
        grid((0, 0), (l, l), step: 1.0, stroke: gray + 0.2pt)
      })
      on-xy({
        line(
          (0, -0.2),
          (0, 0 + 0.2),
          (k, k + 0.2),
          (k, -k - 0.2),
          close: true,
          fill: red.transparentize(50%),
          stroke: none,
        )
        line(
          (0, 0.2),
          (k, k + 0.2),
          (0, k + 0.2),
          close: true,
          fill: gray.transparentize(50%),
          stroke: none,
        )
        line(
          (0, -0.2),
          (k, -k - 0.2),
          (0, -k - 0.2),
          close: true,
          fill: gray.transparentize(50%),
          stroke: none,
        )
        grid((0, -l), (l, l), step: 1.0, stroke: gray + 0.2pt)
        content((k * 0.3, -k * 0.7), [0 (@zero)])
        content((k * 0.7, -k * 0.3), [@init], anchor: "west")
      })
      for i in range(1, k) {
        on-xy(
          {
            line(
              (0, -0.2),
              (0, 0.2),
              (k - i, k - i + 0.2),
              (k - i, -k + i - 0.2),
              close: true,
              fill: blue.transparentize(50%),
              stroke: none,
            )
          },
          z: i,
        )
      }
      on-yz({
        grid((0, -l), (l, l), step: 1.0, stroke: gray + 0.2pt)
      })
      anchor("x2", (1, 1, 2))
      anchor("x3", (2, 2, 1))
      anchor("x4", (0, 2, 1))
      circle("x2", fill: black, radius: 0.1)
      circle("x3", fill: black, radius: 0.1)
      circle("x4", fill: black, radius: 0.1)
      content("x2", [3: $(n',m',n+1)$], anchor: "west")
      content("x3", [1: $(n'+1,m'+1),n)$], anchor: "east")
      content("x4", [2: $(n'-1,m'+1),n)$], anchor: "east")
      content((l, 0, 0), $n'$, anchor: "west")
      content((0, l, 0), $m'$, anchor: "west")
      content((0, 0, l), $- m = n$, anchor: "west")
      line("x3", "x2", mark: (end: ">"))
      line("x4", "x2", mark: (end: ">"))
    },
  )
})

#proof[
  Apply @swap to @b1.
]

#colorbox(
  title: "Calculated",
  color: (
    stroke: blue,
  ),
)[$efc^(m',m)_(n',n)$ for $abs(m') <= n' <= 2 N - n and -m = n <= N$
]

#theorem[4.7-9][
  $
    efc^(m' m)_(n' n) = (-1)^(n + n') efc^(m m')_(n n')
  $
] <swap>

#colorbox(
  title: "Calculated",
  color: (
    stroke: yellow,
  ),
)[
  //- $efc^(m',m)_(n',n)$ for $abs(m') <= n' <= N and abs(m) = n <= N$
  $efc^(m',m)_(n',n)$ for $abs(m) <= n <= 2 N - n' and abs(m') = n' <= N$
]

#theorem[4.26][
  $
    a^m_(n-1) efc^(m' m)_(n' n-1) - a^m_n efc^(m' m)_(n' n+1) = a^m'_n' efc^(m' m)_(n' + 1 n) - a^m'_(n' - 1) efc^(m' m)_(n' - 1 n)
  $
  $
    a^m_n = 1_(abs(m) <= n) sqrt(((n + 1 + abs(m)) (n + 1 - abs(m)))/((2n+1)(2n+3)))
  $
]

#remark[
  $m, m'$ are fixed.
]

$abs(m') < m$

#cetz.canvas({
  import cetz.draw: *
  let l = 13
  let k = l - 0.4
  let md = 1
  let m = 3
  let N = 6
  line((0, 0), (l, 0), mark: (end: ">"))
  line((0, 0), (0, l), mark: (end: ">"))
  content((l, 0), $n$, anchor: "west")
  content((0, l), $n'$, anchor: "south")
  grid((0, 0), (k, k), step: 1.0, stroke: gray + 0.2pt)
  rect((m - 0.2, md - 0.2), (N, N), fill: maroon.transparentize(50%))
  line((m, md), (2 * N - calc.max(m, md), md), stroke: yellow + 2pt)
  line(
    (m, md),
    (2 * N - calc.max(m, md), md),
    (N, N - 2),
    close: true,
    fill: yellow.transparentize(70%),
    stroke: none,
  )
  content((l, md), [$n' = abs(m')$])
  content((0, md), $abs(m')$, anchor: "east")
  line((m, md), (m, 2 * N - calc.max(m, md)), stroke: blue + 2pt)
  line(
    (m, md),
    (m, 2 * N - calc.max(m, md)),
    (N, N),
    (N, N - 2),
    close: true,
    fill: blue.transparentize(70%),
    stroke: none,
  )
  anchor("x1", (6, 2))
  anchor("x2", (7, 3))
  anchor("x3", (7, 1))
  anchor("x4", (8, 2))
  content("x1", "1", anchor: "east")
  content("x2", "3", anchor: "south")
  content("x3", "4", anchor: "west")
  content("x4", "2", anchor: "north")
  line("x1", "x2", mark: (start: "o", end: ">"))
  line("x3", "x2", mark: (start: "o", end: ">"))
  line("x4", "x2", mark: (start: "o", end: ">"))
  anchor("x5", (3, 7))
  anchor("x6", (4, 8))
  anchor("x7", (4, 6))
  anchor("x8", (5, 7))
  content("x5", "1", anchor: "east")
  content("x6", "3", anchor: "south")
  content("x7", "4", anchor: "west")
  content("x8", "2", anchor: "north")
  line("x5", "x8", mark: (start: "o", end: ">"))
  line("x6", "x8", mark: (start: "o", end: ">"))
  line("x7", "x8", mark: (start: "o", end: ">"))
  content((m, l), [$n = abs(m)$])
  content((m, 0), $abs(m)$, anchor: "north")
  content((0, N), $N$, anchor: "east")
  content((0, 2 * N - m), $2N-max(abs(m),abs(m'))$, anchor: "east")
  content((N, 0), $N$, anchor: "north")
  content((2 * N - md, 0), $2N-max(abs(m),abs(m'))$, anchor: "north")
  line(
    (0, 0),
    (l, 0),
    (l, md - 0.2),
    (m - 0.2, md - 0.2),
    (m - 0.2, l),
    (0, l),
    close: true,
    stroke: none,
    fill: gray.transparentize(50%),
  )
  content((m / 2, md / 2), [$0$ (@zero)])
})

#remark[
  All operations above can be performed without using nested for-loop.
  Only $Theta(N)$ recursion is needed and other operations can be done concurrently.
]

= Rotation

Rotation is not related to Helmholtz equation.

$
  T^(m' m)_n (Q) := ip(Y^m_n (theta, phi), Y^m'_n (hat(theta), hat(phi)))
$
$
  Q: SS^2 -> SS^2, (theta, phi) |-> (hat(theta), hat(phi))
$
$
  (theta, phi) := Q(0, 0)
$

$
  Y^m_n (theta, phi) = sum_(abs(m') <= n) T^(m' m)_n (Q) Y^(m')_n (theta', phi')
$

#theorem[
  $
    T^(m' 0)_n = sqrt((4 pi)/(2 n + 1)) Y^(-m')_n (theta', phi')
  $
]
#theorem[

]
