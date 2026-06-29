import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $p$ be an odd prime and let $\mathbb{Z}_p$ denote (the field of) integers modulo $p$. How many elements are in the set $\{x^2:x \in \mathbb{Z}_p\} \cap \{y^2+1:y \in \mathbb{Z}_p\}$?

## Answer
$\lceil p / 4 \rceil$
-/

-- # Formal Answer
noncomputable abbrev Q143.gt := fun (p : ℕ) (answer : ℕ → ℕ) => answer = fun (p : ℕ) => (⌈((↑p : ℝ) / (4 : ℝ) : ℝ)⌉₊ : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q143
  (p : ℕ)
find (answer : ℕ → ℕ) s.t.
  (podd : Odd p)
  (pprime : Prime p)
: (({z : ZMod p | ∃ x : ZMod p, z = x ^ 2} ∩ {z : ZMod p | ∃ y : ZMod p, z = y ^ 2 + 1}).encard = answer p)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q143
  (p : ℕ)
find_all (answer : ℕ → ℕ) iff
  (podd : Odd p)
  (pprime : Prime p)
: (({z : ZMod p | ∃ x : ZMod p, z = x ^ 2} ∩ {z : ZMod p | ∃ y : ZMod p, z = y ^ 2 + 1}).encard = answer p)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
