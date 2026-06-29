import FormalProblemSolving.Basic

/-
## Problem
Given a positive integer $n$, what is the largest $k$ such that the numbers $1,2,\dots,n$ can be put into $k$ boxes so that the sum of the numbers in each box is the same? [When $n=8$, the example $\{1,2,3,6\},\{4,8\},\{5,7\}$ shows that the largest $k$ is \emph{at least} $3$.]

## Answer
the ceiling of n/2
-/

-- # Formal Answer
noncomputable abbrev Q229.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => (⌈((↑n : ℝ) / (2 : ℝ) : ℝ)⌉₊ : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q229
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (kboxes : ℕ → Prop)
  (npos : n > 0)
  (hkboxes : ∀ k : ℕ, kboxes k =
  (∃ boxes : Finset.Icc 1 n → Fin k, ∀ i j : Fin k,
    ∑ x in Finset.univ.filter (boxes · = i), (x : ℕ) =
    ∑ x in Finset.univ.filter (boxes · = j), (x : ℕ)))
: (IsGreatest kboxes (answer n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q229
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (kboxes : ℕ → Prop)
  (npos : n > 0)
  (hkboxes : ∀ k : ℕ, kboxes k =
  (∃ boxes : Finset.Icc 1 n → Fin k, ∀ i j : Fin k,
    ∑ x in Finset.univ.filter (boxes · = i), (x : ℕ) =
    ∑ x in Finset.univ.filter (boxes · = j), (x : ℕ)))
: (IsGreatest kboxes (answer n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
