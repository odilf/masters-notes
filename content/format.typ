#import "@preview/libra:0.1.0": balance
#import "@preview/ctheorems:1.1.3": *

#let preview-args = json(bytes(sys.inputs.at("x-preview", default: "{}")))

#let notes(title, subtitle: none) = it => {  
  set page(paper: "a2")
  set heading(numbering: "1.1)")

  set page(height: auto, width: 15cm, margin: 1cm)
  set page(fill: oklch(23%, 2.5%, 260deg))
  set text(fill: white)
  set table(stroke: white + 0.5pt)
  show pagebreak: it => v(2cm)
  show link: underline

  show math.equation.where(block: false): set text(bottom-edge: "bounds")
  
  balance(text(size: 36pt, tracking: -0.5mm, title))
  if subtitle != none {
    linebreak()
    subtitle
  }
  outline()

  v(1cm)

  // Theorems
  show: thmrules.with(qed-symbol: $square$)

  // Lilaq dark mode
  import "@preview/lilaq:0.4.0" as lq
  show: lq.theme.moon

  it
  v(10cm)
}

#let theorem = thmbox("theorem", "Theorem", fill: oklch(30%, 23%, 310deg))
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
)
#let definition = (..args) => {
  set enum(numbering: "(i)")
  thmbox("definition", "Definition", inset: (x: 1.2em, y: 1em), fill: oklch(30%, 23%, 280deg))(..args)
}
#let example = thmbox("example", "Example", fill: oklch(20%, 0%, 270deg)).with(numbering: none)
#let proof = thmproof("proof", "Proof")

#let todo = body => text(fill: red)[*TODO: #body*]
#let faint(..args) = text(fill: white.transparentize(50%), ..args)

#let dx = $dif x$

This is to have some text when previewing the "format" doc.
