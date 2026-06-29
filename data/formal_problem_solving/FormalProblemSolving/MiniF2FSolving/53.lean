import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Define a sequence recursively by $t_1 = 20$, $t_2 = 21$, and$t_n = \frac{5t_{n-1}+1}{25t_{n-2}}$for all $n \ge 3$. Then $t_{2020}$ can be expressed as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.

## Answer
626
-/

-- # Formal Answer
noncomputable abbrev Q53.gt := fun (answer : ℕ) => answer = (626 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q53
find (answer : ℕ) s.t.
  (p : ℤ)
  (q : ℤ)
  (t : ℕ → ℚ)
  (h₀ : t 1 = 20)
  (h₁ : t 2 = 21)
  (h₂ : ∀ n ≥ 3, t n = (5 * t (n - 1) + 1) / (25 * t (n - 2)))
  (h₃ : t 2020 = (p : ℤ) / (q : ℤ))
  (h₄ : Nat.Coprime p.natAbs q.natAbs)
: (answer = p.natAbs + q.natAbs)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q53
find_all (answer : ℕ) iff
  (p : ℤ)
  (q : ℤ)
  (t : ℕ → ℚ)
  (h₀ : t 1 = 20)
  (h₁ : t 2 = 21)
  (h₂ : ∀ n ≥ 3, t n = (5 * t (n - 1) + 1) / (25 * t (n - 2)))
  (h₃ : t 2020 = (p : ℤ) / (q : ℤ))
  (h₄ : Nat.Coprime p.natAbs q.natAbs)
: (answer = p.natAbs + q.natAbs)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
