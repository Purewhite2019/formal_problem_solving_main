import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let \[f(x) =
\begin{cases}
|\lfloor{x}\rfloor| &\text{if }x\text{ is rational}, \\
\lceil{x}\rceil^2 &\text{if }x\text{ is irrational}.
\end{cases}
\] Find $f(\sqrt[3]{-8})+f(-\pi)+f(\sqrt{50})+f\left(\frac{9}{2}\right)$.

## Answer
79
-/

-- # Formal Answer
noncomputable abbrev Q131.gt := fun (answer : ℤ) => answer = (79 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q131
find (answer : ℤ) s.t.
  (f : ℝ → ℤ)
  (h₀ : ∀ x : ℝ, (¬ Irrational x) → f x = abs (Int.floor x))
  (h₁ : ∀ x : ℝ, (Irrational x) → f x = (Int.ceil x)^2)
: (answer = f (-2) + f (-π) + f (Real.sqrt 50) + f (9/2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q131
find_all (answer : ℤ) iff
  (f : ℝ → ℤ)
  (h₀ : ∀ x : ℝ, (¬ Irrational x) → f x = abs (Int.floor x))
  (h₁ : ∀ x : ℝ, (Irrational x) → f x = (Int.ceil x)^2)
: (answer = f (-2) + f (-π) + f (Real.sqrt 50) + f (9/2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
