import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the value of $a_2+a_4+a_6+a_8+\ldots+a_{98}$ if $a_1$, $a_2$, $a_3\ldots$ is an [[arithmetic progression]] with common difference 1, and $a_1+a_2+a_3+\ldots+a_{98}=137$.

## Answer
93
-/

-- # Formal Answer
noncomputable abbrev Q289.gt := fun (answer : ℚ) => answer = (93 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q289
find (answer : ℚ) s.t.
  (u : ℕ → ℚ)
  (h₀ : ∀ n, u (n + 1) = u n + 1)
  (h₁ : ∑ k in Finset.range 98, u k.succ = 137)
: (answer = ∑ k in Finset.range 49, u (2 * k.succ))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q289
find_all (answer : ℚ) iff
  (u : ℕ → ℚ)
  (h₀ : ∀ n, u (n + 1) = u n + 1)
  (h₁ : ∑ k in Finset.range 98, u k.succ = 137)
: (answer = ∑ k in Finset.range 49, u (2 * k.succ))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
