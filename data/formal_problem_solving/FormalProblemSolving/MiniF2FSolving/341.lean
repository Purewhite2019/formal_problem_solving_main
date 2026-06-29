import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of two natural numbers is $17{,}402$. One of the two numbers is divisible by $10$. If the units digit of that number is erased, the other number is obtained. What is the difference of these two numbers?

$\textbf{(A)} ~10{,}272\qquad\textbf{(B)} ~11{,}700\qquad\textbf{(C)} ~13{,}362\qquad\textbf{(D)} ~14{,}238\qquad\textbf{(E)}

## Answer
14{,}238
-/

-- # Formal Answer
noncomputable abbrev Q341.gt := fun (answer : ℤ) => answer = (14238 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q341
find (answer : ℤ) s.t.
  (x : ℕ)
  (y : ℕ)
  (h₀ : x + y = 17402)
  (h₁ : 10∣x)
  (h₂ : x / 10 = y)
: (answer = ↑x - ↑y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q341
find_all (answer : ℤ) iff
  (x : ℕ)
  (y : ℕ)
  (h₀ : x + y = 17402)
  (h₁ : 10∣x)
  (h₂ : x / 10 = y)
: (answer = ↑x - ↑y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
