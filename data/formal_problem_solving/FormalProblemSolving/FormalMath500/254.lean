import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the average of the two smallest positive integer solutions to the congruence $$14u \equiv 46 \pmod{100}~?$$

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q254.gt := fun (answer : ℝ) => answer = (64 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q254
find (answer : ℝ) s.t.
  (x : ℕ)
  (y : ℕ)
  (S : Set ℕ)
  (hS : S = {u : ℕ | 14 * u ≡ 46 [MOD 100]})
  (hx : IsLeast S x)
  (hy : IsLeast (S\{x}) y)
: (answer = (1:ℝ)*(x + y) / 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q254
find_all (answer : ℝ) iff
  (x : ℕ)
  (y : ℕ)
  (S : Set ℕ)
  (hS : S = {u : ℕ | 14 * u ≡ 46 [MOD 100]})
  (hx : IsLeast S x)
  (hy : IsLeast (S\{x}) y)
: (answer = (1:ℝ)*(x + y) / 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
