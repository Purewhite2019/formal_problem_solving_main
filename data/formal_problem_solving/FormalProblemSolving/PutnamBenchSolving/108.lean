import FormalProblemSolving.Basic
open Real Equiv Polynomial Filter Topology

/-
## Problem
For a positive real number $r$, let $G(r)$ be the minimum value of $|r - \sqrt{m^2+2n^2}|$ for all integers $m$ and $n$. Prove or disprove the assertion that $\lim_{r\to \infty}G(r)$ exists and equals $0$.

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q108.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q108
find (answer : Prop) s.t.
  (G : ℝ → ℝ)
  (hGeq : ∀ r : ℝ, ∃ m n : ℤ, G r = |r - Real.sqrt (m ^ 2 + 2 * n ^ 2)|)
  (hGlb : ∀ r : ℝ, ∀ m n : ℤ, G r ≤ |r - Real.sqrt (m ^ 2 + 2 * n ^ 2)|)
: (answer ↔ Tendsto G atTop (𝓝 0))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q108
find_all (answer : Prop) iff
  (G : ℝ → ℝ)
  (hGeq : ∀ r : ℝ, ∃ m n : ℤ, G r = |r - Real.sqrt (m ^ 2 + 2 * n ^ 2)|)
  (hGlb : ∀ r : ℝ, ∀ m n : ℤ, G r ≤ |r - Real.sqrt (m ^ 2 + 2 * n ^ 2)|)
: (answer ↔ Tendsto G atTop (𝓝 0))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
