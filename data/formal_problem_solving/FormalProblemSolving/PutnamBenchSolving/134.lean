import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
If $X$ is a finite set, let $|X|$ denote the number of elements in $X$. Call an ordered pair $(S,T)$ of subsets of $\{1,2,\dots,n\}$ \emph{admissible} if $s>|T|$ for each $s \in S$, and $t>|S|$ for each $t \in T$. How many admissible ordered pairs of subsets of $\{1,2,\dots,10\}$ are there? Prove your answer.

## Answer
17711
-/

-- # Formal Answer
noncomputable abbrev Q134.gt := fun (answer : ℕ) => answer = (17711 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q134
find (answer : ℕ) s.t.
: (((Finset.univ : Finset <| Finset (Set.Icc 1 10) × Finset (Set.Icc 1 10)).filter
      fun ⟨S, T⟩ ↦ (∀ s ∈ S, T.card < s) ∧ (∀ t ∈ T, S.card < t)).card =
    answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q134
find_all (answer : ℕ) iff
: (((Finset.univ : Finset <| Finset (Set.Icc 1 10) × Finset (Set.Icc 1 10)).filter
      fun ⟨S, T⟩ ↦ (∀ s ∈ S, T.card < s) ∧ (∀ t ∈ T, S.card < t)).card =
    answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
