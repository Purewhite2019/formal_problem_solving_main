import FormalProblemSolving.Basic
open Matrix Function

/-
## Problem
Let $f(x,y)$ be a continuous, real-valued function on $\mathbb{R}^2$. Suppose that, for every rectangular region $R$ of area $1$, the double integral of $f(x,y)$ over $R$ equals $0$.
Must $f(x,y)$ be identically $0$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q244.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q244
find (answer : Prop) s.t.
  (p : ((ℝ × ℝ) → ℝ) → Prop)
  (hp : ∀ f, p f ↔
      Continuous f ∧
      ∀ x1 x2 y1 y2 : ℝ, x2 > x1 → y2 > y1
        → (x2 - x1) * (y2 - y1) = 1 → ∫ x in x1..x2, ∫ y in y1..y2, f (x, y) = 0)
: ((∀ f x y, p f → f (x, y) = 0) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q244
find_all (answer : Prop) iff
  (p : ((ℝ × ℝ) → ℝ) → Prop)
  (hp : ∀ f, p f ↔
      Continuous f ∧
      ∀ x1 x2 y1 y2 : ℝ, x2 > x1 → y2 > y1
        → (x2 - x1) * (y2 - y1) = 1 → ∫ x in x1..x2, ∫ y in y1..y2, f (x, y) = 0)
: ((∀ f x y, p f → f (x, y) = 0) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
