import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The number $a=\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers, has the property that the sum of all real numbers $x$ satisfying
$ \lfloor x \rfloor \cdot \{x\} = a \cdot x^2$
is $420$, where $\lfloor x \rfloor$ denotes the greatest integer less than or equal to $x$ and $\{x\}=x- \lfloor x \rfloor$ denotes the fractional part of $x$. What is $p+q$?

$\textbf{(A) } 245 \qquad \textbf{(B) } 593 \qquad \textbf{(C) } 929 \qquad \textbf{(D) } 1331 \qquad \textbf{(E) } 1332$

## Answer
929
-/

-- # Formal Answer
noncomputable abbrev Q365.gt := fun (answer : ℕ) => answer = (929 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q365
find (answer : ℕ) s.t.
  (a : ℚ)
  (p : ℕ)
  (q : ℕ)
  (h₀ : a = (p : ℚ) / (q : ℚ))
  (h_pos_p : 0 < p)
  (h_pos_q : 0 < q)
  (h₁ : Nat.Coprime p q)
  (S : Finset ℝ)
  (h₂ : ∀ (x : ℝ), x ∈ S ↔ (⌊x⌋ : ℝ) * (x - ⌊x⌋) = (a : ℝ) * x ^ 2)
  (h₃ : ∑ k in S, k = 420)
: (answer = p + q)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q365
find_all (answer : ℕ) iff
  (a : ℚ)
  (p : ℕ)
  (q : ℕ)
  (h₀ : a = (p : ℚ) / (q : ℚ))
  (h_pos_p : 0 < p)
  (h_pos_q : 0 < q)
  (h₁ : Nat.Coprime p q)
  (S : Finset ℝ)
  (h₂ : ∀ (x : ℝ), x ∈ S ↔ (⌊x⌋ : ℝ) * (x - ⌊x⌋) = (a : ℝ) * x ^ 2)
  (h₃ : ∑ k in S, k = 420)
: (answer = p + q)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
