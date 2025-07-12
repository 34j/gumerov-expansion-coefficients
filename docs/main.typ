#import "@preview/algorithmic:0.1.0"
#import "@preview/physica:0.9.2": *
#import "@preview/ctheorems:1.1.3": *
#import "@preview/cetz:0.3.1"

#import algorithmic: algorithm
#let thmboxargs = (inset: 0em, padding: (top: 0em, bottom: 0em))
#let definition = thmbox("definition", "Definition", ..thmboxargs)
#let theorem = thmbox("theorem", "Theorem", ..thmboxargs)
#let lemma = thmbox("lemma", "Lemma", ..thmboxargs)
#let corollary = thmbox("corollary", "Corollary", ..thmboxargs)
#let example = thmbox("example", "Example", ..thmboxargs)
#let axiom = thmbox("axiom", "Axiom", ..thmboxargs)
#let proof = thmproof("proof", "Proof")
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
]
#theorem[2.18-19][
  $
  E^m_n (x + t) = sum_(m', n') efc^(m' m)_(n' n) (t) F^m'_n' (x)
  $
]
#theorem[4.26][
  $
  a^m_(n-1) efc^(m' m)_(n' n-1) - a^m_n efc^(m' m)_(n' n+1) = a^m'_n' efc^(m' m)_(n' + 1 n) - a^m'_(n' - 1) efc^(m' m)_(n' - 1 n)
  $
  $
  a^m_n = 1_(abs(m) <= n) sqrt(((n + 1 + abs(m)) (n + 1 - abs(m)))/((2n+1)(2n+3)))
  $
]
#theorem[4.30][
  $
  b^m_n underbrace(efc^(m' m+1)_(n' n-1),m = n ==> 0 "(4.31)") - b^(-m-1)_(n+1) efc^(m' m+1)_(n' n+1) = b^(m'-1)_(n'+1) efc^(m'-1 m)_(n' + 1 n) - b^(-m')_(n') efc^(m'-1 m)_(n' - 1 n)
  $
  $
  b^m_n = 1_(abs(m) <= n) sign(m) sqrt(((n-m-1)(n-m))/((2n-1)(2n+1)))
  $
]

#cetz.canvas({
  import cetz.draw: *
  let AXIS_LENGTH = 2.8
  ortho(x: 20deg, y: -30deg, z: 0deg, {
    line((0, 0, 0), (0, 0, AXIS_LENGTH), mark: (end: ">"))
    line((0, 0, 0), (0, AXIS_LENGTH, 0), mark: (end: ">"))
    line((0, 0, 0), (AXIS_LENGTH, 0, 0), mark: (end: ">"))
    on-xz({
      grid((0, 0), (AXIS_LENGTH, AXIS_LENGTH), step: 1.0, stroke: gray + 0.2pt)
    })
    on-xy({
      grid((0, 0), (AXIS_LENGTH, AXIS_LENGTH), step: 1.0, stroke: gray + 0.2pt)
    })
    on-yz({
      grid((0, 0), (AXIS_LENGTH, AXIS_LENGTH), step: 1.0, stroke: gray + 0.2pt)
    })
    anchor("x2", (1, 1, 1))
    anchor("x3", (0, 2, 0))
    anchor("x4", (0, 0, 0))
    circle("x2", fill: black, radius: 0.1)
    circle("x3", fill: black, radius: 0.1)
    circle("x4", fill: black, radius: 0.1)
    content("x2", $(m',n',m+1)$, anchor: "west")
    content("x3", $(m'-1,n'+1,m)$, anchor: "east")
    content("x4", $(m'-1,n'-1,m)$, anchor: "east")
    content((AXIS_LENGTH, 0, 0), $m'$, anchor: "west")
    content((0, AXIS_LENGTH, 0), $n'$, anchor: "west")
    content((0, 0, AXIS_LENGTH), $m = n$, anchor: "west")
    line("x3", "x2", mark: (end: ">"))
    line("x4", "x2", mark: (end: ">"))
  })
})
#theorem[4.34][
  $
  b^(m-1)_(n+1) efc^(m' m-1)_(n' n+1) - b^(-m)_n efc^(m' m-1)_(n' n-1) = b^m'_n' efc^(m'+1 m)_(n'-1 n) - b^(-m'-1)_(n'+1) efc^(m'+1 m)_(n'+1 n)
  $
  $
  b^m_n = 1_(abs(m) <= n) sign(m) sqrt(((n-m-1)(n-m))/((2n-1)(2n+1)))
  $
]
#theorem[4.7-9][
  $
  efc^(m' m)_(n' n) = (-1)^(n + n') efc^(m m')_(n n')
  $
]
#theorem[4.43, 4.58][
  $
  efc^(m' 0)_(n' 0) (t) = sqrt(4 pi) (-1)^n cases(R &(E = F),S &(E != F))^(-m')_n' (t)
  $
]
