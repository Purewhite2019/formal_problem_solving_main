import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A sequence of numbers is defined recursively by $a_1 = 1$, $a_2 = \frac{3}{7}$, and
$a_n=\frac{a_{n-2} \cdot a_{n-1}}{2a_{n-2} - a_{n-1}}$for all $n \geq 3$ Then $a_{2019}$ can be written as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. What is $p+q ?$

$\textbf{(A) } 2020 \qquad\textbf{(B) } 4039 \qquad\textbf{(C) } 6057 \qquad\textbf{(D) } 6061 \qquad\textbf{(E) } 8078$

## Answer
8078
-/

-- # Formal Answer
noncomputable abbrev Q11.gt := fun (answer : ℕ) => answer = (8078 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q11
find (answer : ℕ) s.t.
  (a : ℕ → ℚ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 3/7)
  (h₂ : ∀ n : ℕ, 1 ≤ n → a (n+2) = (a n * a (n+1)) / (2 * a n - a (n+1)))
: (answer = (a 2019).den + (a 2019).num.natAbs)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q11
find_all (answer : ℕ) iff
  (a : ℕ → ℚ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 3/7)
  (h₂ : ∀ n : ℕ, 1 ≤ n → a (n+2) = (a n * a (n+1)) / (2 * a n - a (n+1)))
: (answer = (a 2019).den + (a 2019).num.natAbs)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
