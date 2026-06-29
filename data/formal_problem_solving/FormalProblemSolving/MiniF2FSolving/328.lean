import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Four positive integers $a$, $b$, $c$, and $d$ have a product of $8!$ and satisfy:

$
\begin{align*}
ab + a + b & = 524
\\
bc + b + c & = 146
\\
cd + c + d & = 104
\end{align*}
$

What is $a-d$?

$
\text{(A) }4
\qquad
\text{(B) }6
\qquad
\text{(C) }8
\qquad
\text{(D) }10
\qquad
\text{(E) }12
$

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q328.gt := fun (answer : ℤ) => answer = (10 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q328
find (answer : ℤ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (d : ℕ)
  (h₀ : a * b * c * d = 8!)
  (h₁ : a * b + a + b = 524)
  (h₂ : b * c + b + c = 146)
  (h₃ : c * d + c + d = 104)
: (answer = ↑a - ↑d)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q328
find_all (answer : ℤ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (d : ℕ)
  (h₀ : a * b * c * d = 8!)
  (h₁ : a * b + a + b = 524)
  (h₂ : b * c + b + c = 146)
  (h₃ : c * d + c + d = 104)
: (answer = ↑a - ↑d)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
