import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $\sum_{k=1}^{35}\sin 5k=\tan \frac mn,$ where angles are measured in degrees, and $m_{}$ and $n_{}$ are relatively prime positive integers that satisfy $\frac mn<90,$ find $m+n.$

## Answer
177
-/

-- # Formal Answer
noncomputable abbrev Q350.gt := fun (answer : ℕ) => answer = (177 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q350
find (answer : ℕ) s.t.
  (m : ℕ)
  (n : ℕ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h₀ : Nat.Coprime m n)
  (h₁ : ∑ k in Finset.Icc (1 : ℕ) 35, Real.sin ((5 : ℝ) * (k : ℝ) * π / 180) = Real.tan ((m : ℝ) * π / ((n : ℝ) * 180)))
  (h₂ : (m : ℝ) / n < 90)
: (answer = m + n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q350
find_all (answer : ℕ) iff
  (m : ℕ)
  (n : ℕ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h₀ : Nat.Coprime m n)
  (h₁ : ∑ k in Finset.Icc (1 : ℕ) 35, Real.sin ((5 : ℝ) * (k : ℝ) * π / 180) = Real.tan ((m : ℝ) * π / ((n : ℝ) * 180)))
  (h₂ : (m : ℝ) / n < 90)
: (answer = m + n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
