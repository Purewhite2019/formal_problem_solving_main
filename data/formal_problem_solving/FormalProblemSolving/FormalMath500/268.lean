import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The superfactorial $n\$$ is defined as $n\$ = \underbrace{ n!^{{n!}^{{\cdot}^{{\cdot}^{{\cdot}^{n!}}}}}}_{n!}$. What is the units digit of $4\$$?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q268.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q268
find (answer : ℕ) s.t.
  (f : ℕ → ℕ)
  (h_init : f 0 = 1)
  (hf : ∀ k : ℕ, f (k + 1) = (Nat.factorial 4) ^ (f k))
: (answer = f (Nat.factorial 4) % 10)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q268
find_all (answer : ℕ) iff
  (f : ℕ → ℕ)
  (h_init : f 0 = 1)
  (hf : ∀ k : ℕ, f (k + 1) = (Nat.factorial 4) ^ (f k))
: (answer = f (Nat.factorial 4) % 10)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
