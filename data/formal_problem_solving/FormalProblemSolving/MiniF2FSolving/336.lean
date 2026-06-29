import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
All the roots of the polynomial $z^6-10z^5+Az^4+Bz^3+Cz^2+Dz+16$ are positive integers, possibly repeated. What is the value of $B$?

$\textbf{(A) }{-}88 \qquad \textbf{(B) }{-}80 \qquad \textbf{(C) }{-}64 \qquad \textbf{(D) }{-}41\qquad \textbf{(E) }{-}40$

## Answer
-88
-/

-- # Formal Answer
noncomputable abbrev Q336.gt := fun (answer : ℝ) => answer = (-88 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q336
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (f : ℂ → ℂ)
  (h₀ : ∀ z, f z = z^6 - 10 * z^5 + a * z^4 + b * z^3 + c * z^2 + d * z + 16)
  (h₁ : ∀ z, f z = 0 → (z.im = 0 ∧ 0 < z.re ∧ ↑(Int.floor z.re) = z.re))
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q336
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (f : ℂ → ℂ)
  (h₀ : ∀ z, f z = z^6 - 10 * z^5 + a * z^4 + b * z^3 + c * z^2 + d * z + 16)
  (h₁ : ∀ z, f z = 0 → (z.im = 0 ∧ 0 < z.re ∧ ↑(Int.floor z.re) = z.re))
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
