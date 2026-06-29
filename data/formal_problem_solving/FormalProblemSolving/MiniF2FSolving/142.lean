import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the smallest non-negative integer $a$ that satisfies the congruences: \begin{align*}
&a\equiv 2\pmod 3,\\
&a\equiv 4\pmod 5,\\
&a\equiv 6\pmod 7,\\
&a\equiv 8\pmod 9.
\end{align*}

## Answer
314
-/

-- # Formal Answer
noncomputable abbrev Q142.gt := fun (answer : ℕ) => answer = (314 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q142
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {a | a ≡ 2 [MOD 3] ∧ a ≡ 4 [MOD 5] ∧ a ≡ 6 [MOD 7] ∧ a ≡ 8 [MOD 9]})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q142
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {a | a ≡ 2 [MOD 3] ∧ a ≡ 4 [MOD 5] ∧ a ≡ 6 [MOD 7] ∧ a ≡ 8 [MOD 9]})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
