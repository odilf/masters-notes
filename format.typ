#import "@preview/libra:0.1.0": balance
#import "@preview/ctheorems:1.1.3": *
// #import "@preview/frame-it:1.2.0": *


#let preview-args = json(bytes(sys.inputs.at("x-preview", default: "{}")))

#let notes(title, subtitle: none) = it => {  
  set page(paper: "a2")
  set heading(numbering: "1.1)")

  set page(height: auto, width: 15cm, margin: 1cm)
  set page(fill: oklch(23%, 2.5%, 260deg))
  set text(fill: white)
  set table(stroke: white + 0.5pt)
  show pagebreak: none
  show link: underline
  
  balance(text(size: 36pt, tracking: -0.5mm, title))
  if subtitle != none {
    linebreak()
    subtitle
  }
  outline()

  // Theorems
  show: thmrules.with(qed-symbol: $square$)

  it
}

// #let (theorem, corollary, definition, example, proof) = frames(
//   theorem: ("Theorem",),
// )

#let theorem = thmbox("theorem", "Theorem", fill: oklch(30%, 23%, 310deg))
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
)
#let definition = body => {
  set enum(numbering: "(i)")
  thmbox("definition", "Definition", inset: (x: 1.2em, y: 1em), fill: oklch(30%, 23%, 280deg))(body)
}
#let example = thmbox("example", "Example", fill: oklch(20%, 0%, 270deg)).with(numbering: none)
#let proof = thmproof("proof", "Proof")

#let todo = body => [\<TODO: #body\>]

This is to have some text when previewing the "format" doc.
