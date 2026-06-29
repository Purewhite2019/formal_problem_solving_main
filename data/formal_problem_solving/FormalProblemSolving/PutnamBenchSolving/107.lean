import FormalProblemSolving.Basic
open Real Equiv

/-
## Problem
Prove that there are only a finite number of possibilities for the ordered triple $T=(x-y,y-z,z-x)$, where $x,y,z$ are complex numbers satisfying the simultaneous equations
\[
x(x-1)+2yz = y(y-1)+2zx = z(z-1)+2xy,
\]
and list all such triples $T$.

## Answer
{(0, 0, 0), (0, -1, 1), (1, 0, -1), (-1, 1, 0)}
-/

-- # Formal Answer
noncomputable abbrev Q107.gt := fun (answer : Finset (ℂ × ℂ × ℂ)) => answer =
  (Insert.insert (((0 : ℂ), (((0 : ℂ), (0 : ℂ)) : ℂ × ℂ)) : ℂ × ℂ × ℂ)
      (Insert.insert (((0 : ℂ), (((-1 : ℂ), (1 : ℂ)) : ℂ × ℂ)) : ℂ × ℂ × ℂ)
          (Insert.insert (((1 : ℂ), (((0 : ℂ), (-1 : ℂ)) : ℂ × ℂ)) : ℂ × ℂ × ℂ)
              ({(((-1 : ℂ), (((1 : ℂ), (0 : ℂ)) : ℂ × ℂ)) : ℂ × ℂ × ℂ)} : Finset (ℂ × ℂ × ℂ)) :
            Finset (ℂ × ℂ × ℂ)) :
        Finset (ℂ × ℂ × ℂ)) :
    Finset (ℂ × ℂ × ℂ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q107
find (answer : Finset (ℂ × ℂ × ℂ)) s.t.
: ({T : ℂ × ℂ × ℂ | ∃ x y z : ℂ, T = (x - y, y - z, z - x) ∧ x * (x - 1) + 2 * y * z = y * (y - 1) + 2 * z * x ∧ y * (y - 1) + 2 * z * x = z * (z - 1) + 2 * x * y} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q107
find_all (answer : Finset (ℂ × ℂ × ℂ)) iff
: ({T : ℂ × ℂ × ℂ | ∃ x y z : ℂ, T = (x - y, y - z, z - x) ∧ x * (x - 1) + 2 * y * z = y * (y - 1) + 2 * z * x ∧ y * (y - 1) + 2 * z * x = z * (z - 1) + 2 * x * y} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
