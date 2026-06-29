import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $\left(\sum_{k=1}^{20} \log_{5^k} 3^{k^2}\right)\cdot\left(\sum_{k=1}^{100} \log_{9^k} 25^k\right)?$

$\textbf{(A) }21 \qquad \textbf{(B) }100\log_5 3 \qquad \textbf{(C) }200\log_3 5 \qquad \textbf{(D) }2{,}200\qquad \textbf{(E) }21{,}000$

## Answer
21,000
-/

-- # Formal Answer
noncomputable abbrev Q276.gt := fun (answer : ℝ) => answer = (21000 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q276
find (answer : ℝ) s.t.
  (S₁ : Finset ℕ)
  (S₂ : Finset ℕ)
  (hS₁ : S₁ = Finset.Icc 1 20)
  (hS₂ : S₂ = Finset.Icc 1 100)
: (answer = (∑ k in S₁, (Real.logb (5^k) (3^(k^2)))) * (∑ k in S₂, (Real.logb (9^k) (25^k))))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q276
find_all (answer : ℝ) iff
  (S₁ : Finset ℕ)
  (S₂ : Finset ℕ)
  (hS₁ : S₁ = Finset.Icc 1 20)
  (hS₂ : S₂ = Finset.Icc 1 100)
: (answer = (∑ k in S₁, (Real.logb (5^k) (3^(k^2)))) * (∑ k in S₂, (Real.logb (9^k) (25^k))))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
