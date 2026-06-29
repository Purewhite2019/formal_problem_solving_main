import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $r^{}_{}$ is a [[real number]] for which
<div style="text-align:center">$
\left\lfloor r + \frac{19}{100} \right\rfloor + \left\lfloor r + \frac{20}{100} \right\rfloor + \left\lfloor r + \frac{21}{100} \right\rfloor + \cdots + \left\lfloor r + \frac{91}{100} \right\rfloor = 546.
$</div>
Find $\lfloor 100r \rfloor$. (For real $x^{}_{}$, $\lfloor x \rfloor$ is the [[floor function|greatest integer]] less than or equal to $x^{}_{}$.)

## Answer
743
-/

-- # Formal Answer
noncomputable abbrev Q13.gt := fun (answer : ℤ) => answer = (743 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q13
find (answer : ℤ) s.t.
  (r : ℝ)
  (h₀ : ∑ k in Finset.Icc (19 : ℕ) 91, (Int.floor (r + k / 100)) = 546)
: (answer = Int.floor (100 * r))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q13
find_all (answer : ℤ) iff
  (r : ℝ)
  (h₀ : ∑ k in Finset.Icc (19 : ℕ) 91, (Int.floor (r + k / 100)) = 546)
: (answer = Int.floor (100 * r))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
