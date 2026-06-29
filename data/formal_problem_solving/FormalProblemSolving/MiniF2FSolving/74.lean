import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n>1$ is an integer, the notation $a\equiv b\pmod{n}$ means that $(a-b)$ is a multiple of $n$. Find the sum of all possible values of $n$ such that both of the following are true: $171\equiv80\pmod{n}$ and $468\equiv13\pmod{n}$.

## Answer
111
-/

-- # Formal Answer
noncomputable abbrev Q74.gt := fun (answer : ℕ) => answer = (111 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q74
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 2 ≤ n ∧ 171 ≡ 80 [MOD n] ∧ 468 ≡ 13 [MOD n])
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q74
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 2 ≤ n ∧ 171 ≡ 80 [MOD n] ∧ 468 ≡ 13 [MOD n])
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
