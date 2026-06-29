import FormalProblemSolving.Basic
open Set Filter Topology Real

/-
## Problem
Define a sequence $\{a_i\}$ by $a_1=3$ and $a_{i+1}=3^{a_i}$ for $i \geq 1$. Which integers between $00$ and $99$ inclusive occur as the last two digits in the decimal expansion of infinitely many $a_i$?

## Answer
{87}
-/

-- # Formal Answer
noncomputable abbrev Q95.gt := fun (answer : Set (Fin 100)) => answer = ({(87 : Fin (100 : ℕ))} : Set (Fin (100 : ℕ)))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q95
find (answer : Set (Fin 100)) s.t.
  (a : ℕ → ℕ)
  (ha1 : a 1 = 3)
  (ha : ∀ i ≥ 1, a (i + 1) = 3 ^ a i)
: ({k : Fin 100 | ∀ N : ℕ, ∃ i ≥ N, a i % 100 = k} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q95
find_all (answer : Set (Fin 100)) iff
  (a : ℕ → ℕ)
  (ha1 : a 1 = 3)
  (ha : ∀ i ≥ 1, a (i + 1) = 3 ^ a i)
: ({k : Fin 100 | ∀ N : ℕ, ∃ i ≥ N, a i % 100 = k} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
