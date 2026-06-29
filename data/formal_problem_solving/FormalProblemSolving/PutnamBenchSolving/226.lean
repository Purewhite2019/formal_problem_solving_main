import FormalProblemSolving.Basic
open Topology MvPolynomial Filter Set

/-
## Problem
A game involves jumping to the right on the real number line. If $a$ and $b$ are real numbers and $b > a$, the cost of jumping from $a$ to $b$ is $b^3-ab^2$. For what real numbers $c$ can one travel from $0$ to $1$ in a finite number of jumps with total cost exactly $c$?

## Answer
the interval (1/3, 1]
-/

-- # Formal Answer
noncomputable abbrev Q226.gt := fun (answer : Set ℝ) => answer = (Set.Ioc (1 / 3 : ℝ) (1 : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q226
find (answer : Set ℝ) s.t.
: ({c : ℝ | ∃ s : ℕ → ℝ, s 0 = 0 ∧ StrictMono s ∧ (∃ n : ℕ, s n = 1 ∧ ((∑ i in Finset.range n, ((s (i + 1)) ^ 3 - (s i) * (s (i + 1)) ^ 2)) = c))} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q226
find_all (answer : Set ℝ) iff
: ({c : ℝ | ∃ s : ℕ → ℝ, s 0 = 0 ∧ StrictMono s ∧ (∃ n : ℕ, s n = 1 ∧ ((∑ i in Finset.range n, ((s (i + 1)) ^ 3 - (s i) * (s (i + 1)) ^ 2)) = c))} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
