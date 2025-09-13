#import "@preview/libra:0.1.0": balance
#import "@preview/ctheorems:1.1.3": *
#import "@preview/lilaq:0.4.0" as lq

#let preview-args = json(bytes(sys.inputs.at("x-preview", default: "{}")))

#let common-show-rules = it => {
  set page(paper: "a4")
  set heading(numbering: "1.1)")

  set page(height: auto, width: 15cm, margin: 1cm)
  set page(fill: oklch(23%, 2.5%, 260deg))
  set text(fill: white)
  set table(stroke: white + 0.5pt)

  // https://github.com/typst/typst/discussions/2883
  show link: it => {
    if type(it.dest) == str {
      underline(it)
    } else {
      it
    }
  }

  show math.equation.where(block: false): set text(bottom-edge: "bounds")

  // Theorems
  show: thmrules.with(qed-symbol: $square$)

  // Lilaq dark mode
  show: lq.theme.moon

  it
}

#let notes(title, subtitle: none) = it => {
  show: common-show-rules

  show heading.where(
    level: 1,
  ): it => {
    pagebreak(weak: true)
    it
  }

  balance(text(size: 36pt, tracking: -0.5mm, title))
  if subtitle != none {
    linebreak()
    subtitle
  }


  show outline.entry.where(
    level: 1,
  ): set block(above: 1.5em)
  set outline.entry(fill: line(
    length: 100%,
    stroke: white.transparentize(50%) + 0.5pt,
  ))
  show outline.entry: it => link(
    it.element.location(),
    it.indented(it.prefix(), it.inner()),
  )
  outline()

  it
  v(10cm)
}

#let exercises(title) = it => {
  show: common-show-rules

  set heading(numbering: none)

  it
}

#let exercise-counter = counter(<exercise>)
#let exercise(problem, solution) = {
  context {
    let depth = (
      query(selector(<exercise-start>).before(here())).len()
        - query(selector(<exercise-end>).before(here())).len()
    )
    exercise-counter.step(level: depth + 1)
  }

  [#metadata("Exercise start") <exercise-start>]
  rect(
    fill: oklch(30%, 10%, 310deg, 30%),
    radius: 2pt,
    inset: 10pt,
    stroke: white.transparentize(80%),
    width: 100%,
  )[
    #context heading(depth: exercise-counter.get().len())[
      Exercise #numbering("1.1)", ..exercise-counter.get())
    ]
    #problem
  ]
  solution
  [#metadata("Exercise end") <exercise-end>]
}

#let theorem = thmbox("theorem", "Theorem", fill: oklch(30%, 23%, 310deg))
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong,
)
#let definition = (..args) => {
  set enum(numbering: "(i)")
  thmbox("definition", "Definition", inset: (x: 1.2em, y: 1em), fill: oklch(
    30%,
    23%,
    280deg,
  ))(..args)
}
#let example = thmbox("example", "Example", fill: oklch(20%, 0%, 270deg)).with(
  numbering: none,
)
#let proof = thmproof("proof", "Proof")

#let todo = body => text(fill: red)[*#sym.angle.l #body #sym.angle.r*]
#let faint(..args) = text(fill: white.transparentize(50%), ..args)

#let dx = $dif x$
#let lq = lq


#show: exercises[]

This is to have some text when previewing the "format" doc.

#exercise[
  Hello world

  #exercise[
    Nested exercise
  ][
    Nested solution
  ]
][
  Solution unnested
]
