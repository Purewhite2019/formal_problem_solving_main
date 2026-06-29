import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $T$ be the set of all triples $(a,b,c)$ of positive integers for which there exist triangles with side lengths $a,$ $b,$ $c.$  Compute
\[\sum_{(a,b,c) \in T} \frac{2^a}{3^b 5^c}.\]

## Answer
\frac{17}{21}
-/

-- # Formal Answer
noncomputable abbrev Q346.gt := fun (answer : ℝ) => answer = (17 / 21 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q346
find (answer : ℝ) s.t.
: (answer = ∑' (a : PNat) (b : PNat) (c : PNat),
    if (a : ℕ) + (b : ℕ) > (c : ℕ) ∧ (a : ℕ) + (c : ℕ) > (b : ℕ) ∧ (b : ℕ) + (c : ℕ) > (a : ℕ)
    then (2 : ℝ) ^ (a : ℕ) / ((3 : ℝ) ^ (b : ℕ) * (5 : ℝ) ^ (c : ℕ))
    else 0)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q346
find_all (answer : ℝ) iff
: (answer = ∑' (a : PNat) (b : PNat) (c : PNat),
    if (a : ℕ) + (b : ℕ) > (c : ℕ) ∧ (a : ℕ) + (c : ℕ) > (b : ℕ) ∧ (b : ℕ) + (c : ℕ) > (a : ℕ)
    then (2 : ℝ) ^ (a : ℕ) / ((3 : ℝ) ^ (b : ℕ) * (5 : ℝ) ^ (c : ℕ))
    else 0)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
