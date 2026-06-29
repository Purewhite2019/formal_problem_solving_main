import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The [[function]] f is defined on the [[set]] of [[integer]]s and satisfies $f(n)=\begin{cases}
n-3&\mbox{if}\ n\ge 1000\\
f(f(n+5))&\mbox{if}\ n<1000\end{cases}$

Find $f(84)$.

## Answer
997
-/

-- # Formal Answer
noncomputable abbrev Q324.gt := fun (answer : ℤ) => answer = (997 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q324
find (answer : ℤ) s.t.
  (f : ℤ → ℤ)
  (h₀ : ∀ n, 1000 ≤ n → f n = n - 3)
  (h₁ : ∀ n, n < 1000 → f n = f (f (n + 5)))
: (answer = f 84)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q324
find_all (answer : ℤ) iff
  (f : ℤ → ℤ)
  (h₀ : ∀ n, 1000 ≤ n → f n = n - 3)
  (h₁ : ∀ n, n < 1000 → f n = f (f (n + 5)))
: (answer = f 84)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
