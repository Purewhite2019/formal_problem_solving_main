import FormalProblemSolving.Basic
open Nat Topology Filter

/-
## Problem
Basketball star Shanille O'Keal's team statistician keeps track of the number, $S(N)$, of successful free throws she has made in her first $N$ attempts of the season. Early in the season, $S(N)$ was less than $80\%$ of $N$, but by the end of the season, $S(N)$ was more than $80\%$ of $N$. Was there necessarily a moment in between when $S(N)$ was exactly $80\%$ of $N$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q194.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q194
find (answer : Prop) s.t.
  (S : (ℕ → Fin 2) → ℕ → ℝ)
  (hS : ∀ attempts, ∀ N ≥ 1, S attempts N = (∑ i : Fin N, (attempts i).1) / N)
: (answer ↔ (∀ attempts a b,
    (1 ≤ a ∧ a < b ∧ S attempts a < 0.8 ∧ S attempts b > 0.8) →
      (∃ c : ℕ, a < c ∧ c < b ∧ S attempts c = 0.8)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q194
find_all (answer : Prop) iff
  (S : (ℕ → Fin 2) → ℕ → ℝ)
  (hS : ∀ attempts, ∀ N ≥ 1, S attempts N = (∑ i : Fin N, (attempts i).1) / N)
: (answer ↔ (∀ attempts a b,
    (1 ≤ a ∧ a < b ∧ S attempts a < 0.8 ∧ S attempts b > 0.8) →
      (∃ c : ℕ, a < c ∧ c < b ∧ S attempts c = 0.8)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
