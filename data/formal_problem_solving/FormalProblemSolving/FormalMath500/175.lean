import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Daniel works at an electronics store, and he claims that the popularity of a television (measured in number of sales) is inversely proportional to its cost. If 15 customers buy a television that costs $\$$1500, according to Daniel's theory, how many customers would buy a television that costs $\$$2500?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q175.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q175
find (answer : ℕ) s.t.
  (sales₁ : ℕ)
  (sales₂ : ℕ)
  (cost₁ : ℝ)
  (cost₂ : ℝ)
  (h₀ : cost₁ = 1500)
  (h₁ : sales₁ = 15)
  (h₂ : cost₂ = 2500)
  (h₃ : sales₁ * cost₁ = sales₂ * cost₂)
: (sales₂ = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q175
find_all (answer : ℕ) iff
  (sales₁ : ℕ)
  (sales₂ : ℕ)
  (cost₁ : ℝ)
  (cost₂ : ℝ)
  (h₀ : cost₁ = 1500)
  (h₁ : sales₁ = 15)
  (h₂ : cost₂ = 2500)
  (h₃ : sales₁ * cost₁ = sales₂ * cost₂)
: (sales₂ = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
