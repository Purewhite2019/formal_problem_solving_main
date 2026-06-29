import FormalProblemSolving.Basic
open Nat Set MeasureTheory Topology Filter

/-
## Problem
Consider an infinite series whose $n$th term is given by $\pm \frac{1}{n}$, where the actual values of the $\pm$ signs repeat in blocks of $8$ (so the $\frac{1}{9}$ term has the same sign as the $\frac{1}{1}$ term, and so on). Call such a sequence balanced if each block contains four $+$ and four $-$ signs. Prove that being balanced is a sufficient condition for the sequence to converge. Is being balanced also necessary for the sequence to converge?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q25.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q25
find (answer : Prop) s.t.
  (L : List ℝ)
  (hL : L.length = 8 ∧ ∀ i : Fin L.length, L[i] = 1 ∨ L[i] = -1)
  (pluses : ℕ)
  (hpluses : pluses = {i : Fin L.length | L[i] = 1}.ncard)
  (S : ℕ → ℝ)
  (hS : S = fun n : ℕ ↦ ∑ i in Finset.Icc 1 n, L[i % 8]/i)
: (answer ↔ ((∃ l : ℝ, Tendsto S atTop (𝓝 l)) → pluses = 4))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q25
find_all (answer : Prop) iff
  (L : List ℝ)
  (hL : L.length = 8 ∧ ∀ i : Fin L.length, L[i] = 1 ∨ L[i] = -1)
  (pluses : ℕ)
  (hpluses : pluses = {i : Fin L.length | L[i] = 1}.ncard)
  (S : ℕ → ℝ)
  (hS : S = fun n : ℕ ↦ ∑ i in Finset.Icc 1 n, L[i % 8]/i)
: (answer ↔ ((∃ l : ℝ, Tendsto S atTop (𝓝 l)) → pluses = 4))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
