import FormalProblemSolving.Basic
open Filter Topology Real Nat

/-
## Problem
Evaluate \[ \sqrt[8]{2207 - \frac{1}{2207-\frac{1}{2207-\dots}}}. \] Express your answer in the form $\frac{a+b\sqrt{c}}{d}$, where $a,b,c,d$ are integers.

## Answer
⟨3,1,5,2⟩
-/

-- # Formal Answer
noncomputable abbrev Q161.gt := fun (answer : ℤ × ℤ × ℤ × ℤ) => answer = (((3 : ℤ), (((1 : ℤ), (((5 : ℤ), (2 : ℤ)) : ℤ × ℤ)) : ℤ × ℤ × ℤ)) : ℤ × ℤ × ℤ × ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q161
find (answer : ℤ × ℤ × ℤ × ℤ) s.t.
  (contfrac : ℝ)
  (hcontfrac : contfrac = 2207 - 1 / contfrac)
  (hcontfrac' : 1 < contfrac)
: (let ⟨a, b, c, d⟩ := answer;
    contfrac ^ ((1 : ℝ) / 8) = (a + b * sqrt c) / d)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q161
find_all (answer : ℤ × ℤ × ℤ × ℤ) iff
  (contfrac : ℝ)
  (hcontfrac : contfrac = 2207 - 1 / contfrac)
  (hcontfrac' : 1 < contfrac)
: (let ⟨a, b, c, d⟩ := answer;
    contfrac ^ ((1 : ℝ) / 8) = (a + b * sqrt c) / d)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
