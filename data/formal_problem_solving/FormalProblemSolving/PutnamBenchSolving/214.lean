import FormalProblemSolving.Basic
open Set Nat Function

/-
## Problem
Let $x_0 = 1$ and for $n \geq 0$, let $x_{n+1} = 3x_n + \lfloor x_n \sqrt{5} \rfloor$. In particular, $x_1 = 5$, $x_2 = 26$, $x_3 = 136$, $x_4 = 712$. Find a closed-form expression for $x_{2007}$. ($\lfloor a \rfloor$ means the largest integer $\leq a$.)

## Answer
$\frac{2^{2006}}{\sqrt{5}} \left( \left( \frac{1 + \sqrt{5}}{2} \right)^{3997} - \left( \frac{1 + \sqrt{5}}{2} \right)^{-3997} \right)$
-/

-- # Formal Answer
noncomputable abbrev Q214.gt := fun (answer : ℝ) => answer =
  ((((2 : ℝ) ^ (2006 : ℕ) : ℝ) / (√(5 : ℝ) : ℝ) : ℝ) *
      (((((1 : ℝ) + (√(5 : ℝ) : ℝ) : ℝ) / (2 : ℝ) : ℝ) ^ (3997 : ℕ) : ℝ) -
          ((((1 : ℝ) + (√(5 : ℝ) : ℝ) : ℝ) / (2 : ℝ) : ℝ) ^ (-3997 : ℤ) : ℝ) :
        ℝ) :
    ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q214
find (answer : ℝ) s.t.
  (x : ℕ → ℝ)
  (hx0 : x 0 = 1)
  (hx : ∀ n : ℕ, x (n + 1) = 3 * (x n) + ⌊(x n) * Real.sqrt 5⌋)
: (x 2007 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q214
find_all (answer : ℝ) iff
  (x : ℕ → ℝ)
  (hx0 : x 0 = 1)
  (hx : ∀ n : ℕ, x (n + 1) = 3 * (x n) + ⌊(x n) * Real.sqrt 5⌋)
: (x 2007 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
