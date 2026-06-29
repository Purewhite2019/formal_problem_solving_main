import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A book is said to have $n$ leaves if it is composed of $n$ pieces of paper. On the other hand, the number of pages is twice the number of leaves because each side of a piece of paper is defined as a page.

If the number of pages in a book is $3$ more than a multiple of $7$, and the number of leaves is greater than $100$, then what is the smallest possible number of leaves?

## Answer
103
-/

-- # Formal Answer
noncomputable abbrev Q241.gt := fun (answer : ℕ) => answer = (103 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q241
find (answer : ℕ) s.t.
: (answer > 100) ∧ (2 * answer ≡ 3 [MOD 7]) ∧ (∀ n, n > 100 → 2 * n ≡ 3 [MOD 7] → answer ≤ n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q241
find_all (answer : ℕ) iff
: (answer > 100) ∧ (2 * answer ≡ 3 [MOD 7]) ∧ (∀ n, n > 100 → 2 * n ≡ 3 [MOD 7] → answer ≤ n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
