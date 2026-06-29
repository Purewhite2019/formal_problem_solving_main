import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ and $c$ be positive real numbers.  Find the minimum value of
\[\frac{(a + b + c)[(a + b)^2 + (a + b + 4c)^2]}{abc}.\]

## Answer
100
-/

-- # Formal Answer
noncomputable abbrev Q349.gt := fun (answer : ℝ) => answer = (100 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q349
find (answer : ℝ) s.t.
: (∀ (a b c : ℝ),
    a > (0 : ℝ) →
      b > (0 : ℝ) →
        c > (0 : ℝ) →
          ((((a + b : ℝ) + c : ℝ) *
                    (((a + b : ℝ) ^ (2 : ℕ) : ℝ) + (((a + b : ℝ) + ((4 : ℝ) * c : ℝ) : ℝ) ^ (2 : ℕ) : ℝ) : ℝ) :
                  ℝ) /
                ((a * b : ℝ) * c : ℝ) :
              ℝ) ≥
            answer) ∧
  ∃ (a : ℝ) (b : ℝ) (c : ℝ),
    a > (0 : ℝ) ∧
      b > (0 : ℝ) ∧
        c > (0 : ℝ) ∧
          ((((a + b : ℝ) + c : ℝ) *
                    (((a + b : ℝ) ^ (2 : ℕ) : ℝ) + (((a + b : ℝ) + ((4 : ℝ) * c : ℝ) : ℝ) ^ (2 : ℕ) : ℝ) : ℝ) :
                  ℝ) /
                ((a * b : ℝ) * c : ℝ) :
              ℝ) =
            answer
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q349
find_all (answer : ℝ) iff
: (∀ (a b c : ℝ),
    a > (0 : ℝ) →
      b > (0 : ℝ) →
        c > (0 : ℝ) →
          ((((a + b : ℝ) + c : ℝ) *
                    (((a + b : ℝ) ^ (2 : ℕ) : ℝ) + (((a + b : ℝ) + ((4 : ℝ) * c : ℝ) : ℝ) ^ (2 : ℕ) : ℝ) : ℝ) :
                  ℝ) /
                ((a * b : ℝ) * c : ℝ) :
              ℝ) ≥
            answer) ∧
  ∃ (a : ℝ) (b : ℝ) (c : ℝ),
    a > (0 : ℝ) ∧
      b > (0 : ℝ) ∧
        c > (0 : ℝ) ∧
          ((((a + b : ℝ) + c : ℝ) *
                    (((a + b : ℝ) ^ (2 : ℕ) : ℝ) + (((a + b : ℝ) + ((4 : ℝ) * c : ℝ) : ℝ) ^ (2 : ℕ) : ℝ) : ℝ) :
                  ℝ) /
                ((a * b : ℝ) * c : ℝ) :
              ℝ) =
            answer
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
