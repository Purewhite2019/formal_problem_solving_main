import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,b$ be two natural numbers. When we divide $a^2+b^2$ by $a+b$, we the the remainder $r$ and the quotient $q.$ Determine all pairs $(a, b)$ for which $q^2 + r = 1977.$

## Answer
(a,b)=(37,50) , (7, 50)
-/

-- # Formal Answer
noncomputable abbrev Q170.gt := fun (answer : Set (ℕ × ℕ)) => answer = (Insert.insert (((37 : ℕ), (50 : ℕ)) : ℕ × ℕ) ({(((7 : ℕ), (50 : ℕ)) : ℕ × ℕ)} : Set (ℕ × ℕ)) : Set (ℕ × ℕ))

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q170
find (answer : Set (ℕ × ℕ)) s.t.
: (answer = { (a, b) | ∃ (q r : ℕ), r < a + b ∧ a ^ 2 + b ^ 2 = (a + b) * q + r ∧ q ^ 2 + r = 1977 ∧ a ≤ b })
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q170
find_all (answer : Set (ℕ × ℕ)) iff
: (answer = { (a, b) | ∃ (q r : ℕ), r < a + b ∧ a ^ 2 + b ^ 2 = (a + b) * q + r ∧ q ^ 2 + r = 1977 ∧ a ≤ b })
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
