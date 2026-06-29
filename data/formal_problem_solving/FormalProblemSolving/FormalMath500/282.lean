import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the unique $\textbf{odd}$ integer $t$ such that $0<t<23$ and $t+2$ is the inverse of $t$ modulo $23$.

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q282.gt := fun (answer : ℤ) => answer = (17 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q282
find (answer : ℤ) s.t.
  (t : ℤ)
  (h₀ : 0 < t)
  (h₁ : t < 23)
  (h₂ : t % 2 = 1)
  (h₃ : (t + 2) * t ≡ 1 [ZMOD 23])
: (answer = t)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q282
find_all (answer : ℤ) iff
  (t : ℤ)
  (h₀ : 0 < t)
  (h₁ : t < 23)
  (h₂ : t % 2 = 1)
  (h₃ : (t + 2) * t ≡ 1 [ZMOD 23])
: (answer = t)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
