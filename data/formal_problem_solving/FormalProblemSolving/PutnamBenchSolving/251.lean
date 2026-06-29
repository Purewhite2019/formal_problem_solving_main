import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Suppose $X$ is a random variable that takes on only nonnegative integer values, with $E\left[X\right]=1$, $E\left[X^2\right]=2$, and $E\left[X^3\right]=5$. (Here $E\left[Y\right]$ denotes the expectation of the random variable $Y$.) Determine the smallest possible value of the probability of the event $X=0$.

## Answer
1/3
-/

-- # Formal Answer
noncomputable abbrev Q251.gt := fun (answer : ℝ) => answer = (1 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q251
find (answer : ℝ) s.t.
  (Xrandvar : (ℕ → ℝ) → Prop)
  (hXrandvar : Xrandvar = (fun X : ℕ → ℝ => (∀ n : ℕ, X n ∈ Set.Icc 0 1) ∧ ∑' n : ℕ, X n = 1))
  (E : (ℕ → ℝ) → (ℕ → ℝ) → ℝ)
  (hE : E = (fun (X : ℕ → ℝ) (f : ℕ → ℝ) => ∑' n : ℕ, f n * X n))
: (sInf {X0 : ℝ | ∃ X : ℕ → ℝ, Xrandvar X ∧ E X (fun x : ℕ => x) = 1 ∧ E X (fun x : ℕ => x ^ 2) = 2 ∧ E X (fun x : ℕ => x ^ 3) = 5 ∧ X0 = X 0} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q251
find_all (answer : ℝ) iff
  (Xrandvar : (ℕ → ℝ) → Prop)
  (hXrandvar : Xrandvar = (fun X : ℕ → ℝ => (∀ n : ℕ, X n ∈ Set.Icc 0 1) ∧ ∑' n : ℕ, X n = 1))
  (E : (ℕ → ℝ) → (ℕ → ℝ) → ℝ)
  (hE : E = (fun (X : ℕ → ℝ) (f : ℕ → ℝ) => ∑' n : ℕ, f n * X n))
: (sInf {X0 : ℝ | ∃ X : ℕ → ℝ, Xrandvar X ∧ E X (fun x : ℕ => x) = 1 ∧ E X (fun x : ℕ => x ^ 2) = 2 ∧ E X (fun x : ℕ => x ^ 3) = 5 ∧ X0 = X 0} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
