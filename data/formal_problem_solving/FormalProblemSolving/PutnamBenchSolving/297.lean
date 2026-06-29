import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
A grasshopper starts at the origin in the coordinate plane and makes a sequence of hops.
Each hop has length $5$, and after each hop the grasshopper is at a point whose coordinates are both integers; thus, there are $12$ possible locations for the grasshopper after the first hop.
What is the smallest number of hops needed for the grasshopper to reach the point $(2021, 2021)$?

## Answer
578
-/

-- # Formal Answer
noncomputable abbrev Q297.gt := fun (answer : ℕ) => answer = (578 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q297
find (answer : ℕ) s.t.
  (P : List (ℤ × ℤ) → Prop)
  (P_def : ∀ l, P l ↔ l.Chain' fun p q ↦ (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = 25)
: (IsLeast
      {k | ∃ l, P ((0, 0) :: l) ∧ l.getLast! = (2021, 2021) ∧ l.length = k}
      answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q297
find_all (answer : ℕ) iff
  (P : List (ℤ × ℤ) → Prop)
  (P_def : ∀ l, P l ↔ l.Chain' fun p q ↦ (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = 25)
: (IsLeast
      {k | ∃ l, P ((0, 0) :: l) ∧ l.getLast! = (2021, 2021) ∧ l.length = k}
      answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
