#import "@preview/libra:0.1.0": balance
#import "@preview/ctheorems:1.1.3": *
#import "@preview/lilaq:0.5.0" as lq

#let preview-args = json(bytes(sys.inputs.at("x-preview", default: "{}")))
#let dark-theme = sys.inputs.at("DARK_THEME", default: "false") == "true";

#let common-show-rules = it => {
  set page(paper: "a4")
  set heading(numbering: "1.1)")

  // set page(height: 50cm, width: 15cm, margin: 1cm, numbering: "1")
  if dark-theme {
    set page(fill: oklch(23%, 2.5%, 260deg))
    set text(fill: white)
    set table(stroke: white + 0.5pt)
  }

  // https://github.com/typst/typst/discussions/2883
  show link: it => {
    if type(it.dest) == str {
      underline(it)
    } else {
      it
    }
  }

  show math.equation.where(block: false): set text(bottom-edge: "bounds")
  show math.equation: set block(breakable: true)

  // Theorems
  show: thmrules.with(qed-symbol: $square$)

  if dark-theme {
    show: lq.theme.moon
  }
  show: lq.set-diagram(width: 100%, height: 6cm)

  it
}

#let notes(title, subtitle: none) = it => {
  show: common-show-rules

  show outline.entry.where(
    level: 1,
  ): set block(above: 1.5em)
  if dark-theme {
    set outline.entry(fill: line(
      length: 100%,
      stroke: white.transparentize(50%) + 0.5pt,
    ))
  }
  show outline.entry: it => link(
    it.element.location(),
    it.indented(it.prefix(), it.inner()),
  )

  align(horizon, {
    balance(text(size: 36pt, tracking: -0.5mm, title))
    v(0cm, weak: true)
    if subtitle != none {
      linebreak()
      subtitle
    }

    v(5mm)
    outline()
  })
  show heading: it => {
    if it.level == 1 {
      pagebreak(weak: true)
    } else if it.level == 2 {
      v(1cm)
    }
    it
  }
  it
  v(10cm)
}

#let lm = if dark-theme { 0 } else { 1 }
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
    fill: oklch(30% + lm * 40%, 10%, 310deg, 30%),
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

#set enum(numbering: "(i)")
#let theorem(..args) = {
  thmbox("theorem", "Theorem", fill: oklch(
    30% + lm * 60%,
    23%,
    310deg,
  ))(..args)
}
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong,
)
#let definition = (..args) => {
  set enum(numbering: "(i)")
  thmbox("definition", "Definition", inset: (x: 1.2em, y: 1em), fill: oklch(
    30% + lm * 55%,
    23%,
    280deg,
  ))(..args)
}
#let example = thmplain("example", "Example", fill: oklch(
  20% + lm * 70%,
  0%,
  270deg,
)).with(
  numbering: none,
  inset: 5mm,
)
#let proof = thmproof("proof", "Proof")
#let lemma = thmbox("lemma", "Lemma")


#let lq = lq

// Convenience functions
#let todo = body => text(fill: red)[*#sym.chevron.l #body #sym.chevron.r*]
#let faint(..args) = text(
  fill: if dark-theme { white } else { black }.transparentize(50%),
  ..args,
)

// Convenience math definitions
#let dx = $dif x$
#let innerproduct(x, y) = $lr(chevron.l #x, #y chevron.r)$


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
