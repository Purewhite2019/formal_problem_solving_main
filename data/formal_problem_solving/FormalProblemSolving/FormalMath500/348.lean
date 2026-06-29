import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For every positive integer $n$, let $\text{mod}_5 (n)$ be the remainder obtained when $n$ is divided by 5. Define a function $f: \{0,1,2,3,\dots\} \times \{0,1,2,3,4\} \to \{0,1,2,3,4\}$ recursively as follows:

\[f(i,j) = \begin{cases}\text{mod}_5 (j+1) & \text{ if } i = 0 \text{ and } 0 \le j \le 4 \text{,}\\ f(i-1,1) & \text{ if } i \ge 1 \text{ and } j = 0 \text{, and} \\ f(i-1, f(i,j-1)) & \text{ if } i \ge 1 \text{ and } 1 \le j \le 4. \end{cases}\]What is $f(2015,2)$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q348.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q348
find (answer : ℕ) s.t.
  (f : (ℕ × ℕ) → ℕ)
  (hf : ∀ x y : ℕ, (x = 0 ∧ y ≥ 0 ∧ y ≤ 4) → f (x,y) = (y + 1) % 5)
  (hf2 : ∀ x y : ℕ, (x ≥ 1 ∧ y = 0) → f (x,y) = f (x-1, 1))
  (hf3 : ∀ x y : ℕ, (x ≥ 1 ∧ y ≥ 1 ∧ y ≤ 4) → f (x,y) = f (x-1, f (x,y-1)))
: (answer = f (2015,2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q348
find_all (answer : ℕ) iff
  (f : (ℕ × ℕ) → ℕ)
  (hf : ∀ x y : ℕ, (x = 0 ∧ y ≥ 0 ∧ y ≤ 4) → f (x,y) = (y + 1) % 5)
  (hf2 : ∀ x y : ℕ, (x ≥ 1 ∧ y = 0) → f (x,y) = f (x-1, 1))
  (hf3 : ∀ x y : ℕ, (x ≥ 1 ∧ y ≥ 1 ∧ y ≤ 4) → f (x,y) = f (x-1, f (x,y-1)))
: (answer = f (2015,2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
