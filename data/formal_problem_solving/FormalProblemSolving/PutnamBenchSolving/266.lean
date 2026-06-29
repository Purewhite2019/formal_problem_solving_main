import FormalProblemSolving.Basic
open Polynomial Filter Topology Real Set Nat List

/-
## Problem
Find the smallest constant $C$ such that for every real polynomial $P(x)$ of degree $3$ that has a root in the interval $[0,1]$,
\[
\int_0^1 \left| P(x) \right|\,dx \leq C \max_{x \in [0,1]} \left| P(x) \right|.
\]

## Answer
5 / 6
-/

-- # Formal Answer
noncomputable abbrev Q266.gt := fun (answer : ℝ) => answer = (5 / 6 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q266
find (answer : ℝ) s.t.
  (p : ℝ → Prop)
  (hp : ∀ c, p c ↔
        ∀ P : Polynomial ℝ, P.degree = 3 →
        (∃ x ∈ Icc 0 1, P.eval x = 0) →
        ∫ x in (0)..1, |P.eval x| ≤ c * (sSup {y | ∃ x ∈ Icc 0 1, y = |P.eval x|}))
: (IsLeast p answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q266
find_all (answer : ℝ) iff
  (p : ℝ → Prop)
  (hp : ∀ c, p c ↔
        ∀ P : Polynomial ℝ, P.degree = 3 →
        (∃ x ∈ Icc 0 1, P.eval x = 0) →
        ∫ x in (0)..1, |P.eval x| ≤ c * (sSup {y | ∃ x ∈ Icc 0 1, y = |P.eval x|}))
: (IsLeast p answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
