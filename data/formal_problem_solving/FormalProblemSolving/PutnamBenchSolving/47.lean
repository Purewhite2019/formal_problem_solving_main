import FormalProblemSolving.Basic

/-
## Problem
Let $f, g, h$ be functions $\mathbb{R} \to \mathbb{R}$. Find an expression for $h(x)$ in terms of $f$ and $g$ such that $f(x) = \frac{h(x + 1) + h(x - 1)}{2}$ and $g(x) = \frac{h(x + 4) + h(x - 4)}{2}$.

## Answer
$h(x) = g(x) - f(x - 3) + f(x - 1) + f(x + 1) - f(x + 3)$
-/

-- # Formal Answer
noncomputable abbrev Q47.gt := fun (f : ℝ → ℝ) (g : ℝ → ℝ) (answer : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) => answer = fun (f g : ℝ → ℝ) (x : ℝ) =>
  (((((g x : ℝ) - (f (x - (3 : ℝ) : ℝ) : ℝ) : ℝ) + (f (x - (1 : ℝ) : ℝ) : ℝ) : ℝ) + (f (x + (1 : ℝ) : ℝ) : ℝ) : ℝ) -
      (f (x + (3 : ℝ) : ℝ) : ℝ) :
    ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q47
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
find (answer : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) s.t.
  (h : ℝ → ℝ)
  (hf : ∀ x, f x = (h (x + 1) + h (x - 1)) / 2)
  (hg : ∀ x, g x = (h (x + 4) + h (x - 4)) / 2)
: (h = answer f g)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q47
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
find_all (answer : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) iff
  (h : ℝ → ℝ)
  (hf : ∀ x, f x = (h (x + 1) + h (x - 1)) / 2)
  (hg : ∀ x, g x = (h (x + 4) + h (x - 4)) / 2)
: (h = answer f g)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
