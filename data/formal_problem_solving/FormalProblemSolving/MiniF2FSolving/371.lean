import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $(1+\sin t)(1+\cos t)=5/4$ and
:$(1-\sin t)(1-\cos t)=\frac mn-\sqrt{k},$
where $k, m,$ and $n_{}$ are [[positive integer]]s with $m_{}$ and $n_{}$ [[relatively prime]], find $k+m+n.$

## Answer
027
-/

-- # Formal Answer
noncomputable abbrev Q371.gt := fun (answer : ℕ) => answer = (27 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q371
find (answer : ℕ) s.t.
  (k : ℕ)
  (m : ℕ)
  (n : ℕ)
  (t : ℝ)
  (h₀ : 0 < k ∧ 0 < m ∧ 0 < n)
  (h₁ : Nat.gcd m n = 1)
  (h₂ : (1 + Real.sin t) * (1 + Real.cos t) = 5/4)
  (h₃ : (1 - Real.sin t) * (1 - Real.cos t) = (m : ℝ)/(n : ℝ) - Real.sqrt k)
: (answer = k + m + n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q371
find_all (answer : ℕ) iff
  (k : ℕ)
  (m : ℕ)
  (n : ℕ)
  (t : ℝ)
  (h₀ : 0 < k ∧ 0 < m ∧ 0 < n)
  (h₁ : Nat.gcd m n = 1)
  (h₂ : (1 + Real.sin t) * (1 + Real.cos t) = 5/4)
  (h₃ : (1 - Real.sin t) * (1 - Real.cos t) = (m : ℝ)/(n : ℝ) - Real.sqrt k)
: (answer = k + m + n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
