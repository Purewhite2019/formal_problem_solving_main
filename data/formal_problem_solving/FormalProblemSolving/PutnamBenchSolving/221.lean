import FormalProblemSolving.Basic
open Topology MvPolynomial Filter

/-
## Problem
Let $f$ be a real-valued function on the plane such that for every square $ABCD$ in the plane, $f(A)+f(B)+f(C)+f(D)=0$. Does it follow that $f(P)=0$ for all points $P$ in the plane?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q221.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q221
find (answer : Prop) s.t.
  (f : ℝ × ℝ → ℝ)
  (h_square : ∀ O v : ℝ × ℝ, v ≠ (0, 0) → f (O.1, O.2) + f (O.1 + v.1, O.2 + v.2) + f (O.1 + v.1 - v.2, O.2 + v.2 + v.1) + f (O.1 - v.2, O.2 + v.1) = 0)
: ((∀ P : ℝ × ℝ, f P = 0) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q221
find_all (answer : Prop) iff
  (f : ℝ × ℝ → ℝ)
  (h_square : ∀ O v : ℝ × ℝ, v ≠ (0, 0) → f (O.1, O.2) + f (O.1 + v.1, O.2 + v.2) + f (O.1 + v.1 - v.2, O.2 + v.2 + v.1) + f (O.1 - v.2, O.2 + v.1) = 0)
: ((∀ P : ℝ × ℝ, f P = 0) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
