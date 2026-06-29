import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the first $n$ terms in the infinite geometric sequence $\left\{\frac{1}{4},\frac{1}{8},\frac{1}{16},\dots \right\}$ is $\frac{255}{512}$. Find $n$.

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q128.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q128
find (answer : ℕ) s.t.
: (∑ i in Finset.range answer, (1 : ℝ) / (4 * 2 ^ i) = (255 : ℝ) / 512)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q128
find_all (answer : ℕ) iff
: (∑ i in Finset.range answer, (1 : ℝ) / (4 * 2 ^ i) = (255 : ℝ) / 512)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
