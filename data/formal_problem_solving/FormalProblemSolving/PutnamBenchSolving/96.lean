import FormalProblemSolving.Basic
open Set Filter Topology Real

/-
## Problem
Let $I_m = \int_0^{2\pi} \cos(x)\cos(2x)\cdots \cos(mx)\,dx$. For which integers $m$, $1 \leq m \leq 10$ is $I_m \neq 0$?

## Answer
{3, 4, 7, 8}
-/

-- # Formal Answer
noncomputable abbrev Q96.gt := fun (answer : Set ℕ) => answer =
  (Insert.insert (3 : ℕ) (Insert.insert (4 : ℕ) (Insert.insert (7 : ℕ) ({(8 : ℕ)} : Set ℕ) : Set ℕ) : Set ℕ) : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q96
find (answer : Set ℕ) s.t.
  (I : ℕ → ℝ)
  (hI : I = fun (m : ℕ) ↦ ∫ x in (0)..(2 * Real.pi), ∏ k in Finset.Icc 1 m, cos (k * x))
: ({m ∈ Finset.Icc 1 10 | I m ≠ 0} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q96
find_all (answer : Set ℕ) iff
  (I : ℕ → ℝ)
  (hI : I = fun (m : ℕ) ↦ ∫ x in (0)..(2 * Real.pi), ∏ k in Finset.Icc 1 m, cos (k * x))
: ({m ∈ Finset.Icc 1 10 | I m ≠ 0} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
