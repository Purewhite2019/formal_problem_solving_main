import FormalProblemSolving.Basic

/-
## Problem
Compute $\log_2 \left( \prod_{a=1}^{2015}\prod_{b=1}^{2015}(1+e^{2\pi iab/2015}) \right)$. Here $i$ is the imaginary unit (that is, $i^2=-1$).

## Answer
13725
-/

-- # Formal Answer
noncomputable abbrev Q256.gt := fun (answer : ℂ) => answer = (13725 : ℂ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q256
find (answer : ℂ) s.t.
: (Complex.log (∏ a : Fin 2015, ∏ b : Fin 2015, (1 + Complex.exp (2 * Real.pi * Complex.I * (a.1 + 1) * (b.1 + 1) / 2015))) / Complex.log 2 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q256
find_all (answer : ℂ) iff
: (Complex.log (∏ a : Fin 2015, ∏ b : Fin 2015, (1 + Complex.exp (2 * Real.pi * Complex.I * (a.1 + 1) * (b.1 + 1) / 2015))) / Complex.log 2 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
