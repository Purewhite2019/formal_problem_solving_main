import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $a$ and $b$ be positive numbers. Find the largest number $c$, in terms of $a$ and $b$, such that $a^xb^{1-x} \leq a\frac{\sinh ux}{\sinh u}+b\frac{\sinh u(1-x)}{\sinh u}$ for all $u$ with $0<|u| \leq c$ and for all $x$, $0<x<1$. (Note: $\sinh u=(e^u-e^{-u})/2$.)

## Answer
(fun a b : ℝ => |Real.log (a / b)|)
-/

-- # Formal Answer
noncomputable abbrev Q144.gt := fun (a : ℝ) (b : ℝ) (answer : ℝ → ℝ → ℝ) => answer = fun (a b : ℝ) => (|(Real.log (a / b : ℝ) : ℝ)| : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q144
  (a : ℝ)
  (b : ℝ)
find (answer : ℝ → ℝ → ℝ) s.t.
  (abpos : a > 0 ∧ b > 0)
: (IsGreatest {c | ∀ u, (0 < |u| ∧ |u| ≤ c) → (∀ x ∈ Set.Ioo 0 1, a ^ x * b ^ (1 - x) ≤ a * (Real.sinh (u * x) / Real.sinh u) + b * (Real.sinh (u * (1 - x)) / Real.sinh u))} (answer a b))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q144
  (a : ℝ)
  (b : ℝ)
find_all (answer : ℝ → ℝ → ℝ) iff
  (abpos : a > 0 ∧ b > 0)
: (IsGreatest {c | ∀ u, (0 < |u| ∧ |u| ≤ c) → (∀ x ∈ Set.Ioo 0 1, a ^ x * b ^ (1 - x) ≤ a * (Real.sinh (u * x) / Real.sinh u) + b * (Real.sinh (u * (1 - x)) / Real.sinh u))} (answer a b))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
