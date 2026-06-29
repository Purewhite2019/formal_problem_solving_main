import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the largest $x$-value at which the graphs of $f(x)=e^{3x^2-|\lfloor x \rfloor|!}+\binom{22+735235|\lfloor x \rfloor |}{2356}+\phi(|\lfloor x \rfloor|+1)+72x^4+3x^3-6x^2+2x+1$ and $g(x)=e^{3x^2-|\lfloor x \rfloor|!}+\binom{22+735235|\lfloor x \rfloor |}{2356}+\phi(|\lfloor x \rfloor|+1)+72x^4+4x^3-11x^2-6x+13$ intersect, where $\lfloor x \rfloor$ denotes the floor function of $x$, and $\phi(n)$ denotes the sum of the positive integers $\le$ and relatively prime to $n$.

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q122.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q122
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x : ℝ, f x = Real.exp (3 * x ^ 2 - (Nat.factorial ((Int.floor x).natAbs)))
    + (Nat.choose (22 + 735235 * (Int.floor x).natAbs) 2356 : ℝ)
    + (∑ k in ((Finset.Icc 1 ((Int.floor x).natAbs + 1)).filter fun k => Nat.Coprime k ((Int.floor x).natAbs + 1)), (k : ℝ))
    + 72 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + 2 * x + 1)
  (hg : ∀ x : ℝ, g x = Real.exp (3 * x ^ 2 - (Nat.factorial ((Int.floor x).natAbs)))
    + (Nat.choose (22 + 735235 * (Int.floor x).natAbs) 2356 : ℝ)
    + (∑ k in ((Finset.Icc 1 ((Int.floor x).natAbs + 1)).filter fun k => Nat.Coprime k ((Int.floor x).natAbs + 1)), (k : ℝ))
    + 72 * x ^ 4 + 4 * x ^ 3 - 11 * x ^ 2 - 6 * x + 13)
  (hS : S = {x : ℝ | f x = g x})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q122
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x : ℝ, f x = Real.exp (3 * x ^ 2 - (Nat.factorial ((Int.floor x).natAbs)))
    + (Nat.choose (22 + 735235 * (Int.floor x).natAbs) 2356 : ℝ)
    + (∑ k in ((Finset.Icc 1 ((Int.floor x).natAbs + 1)).filter fun k => Nat.Coprime k ((Int.floor x).natAbs + 1)), (k : ℝ))
    + 72 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + 2 * x + 1)
  (hg : ∀ x : ℝ, g x = Real.exp (3 * x ^ 2 - (Nat.factorial ((Int.floor x).natAbs)))
    + (Nat.choose (22 + 735235 * (Int.floor x).natAbs) 2356 : ℝ)
    + (∑ k in ((Finset.Icc 1 ((Int.floor x).natAbs + 1)).filter fun k => Nat.Coprime k ((Int.floor x).natAbs + 1)), (k : ℝ))
    + 72 * x ^ 4 + 4 * x ^ 3 - 11 * x ^ 2 - 6 * x + 13)
  (hS : S = {x : ℝ | f x = g x})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
