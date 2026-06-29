import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $F(z)=\frac{z+i}{z-i}$ for all complex numbers $z\not= i,$ and let $z_n=F(z_{n-1})$ for all positive integers $n.$ Given that $z_0=\frac 1{137}+i,$ find $z_{2002}.$

## Answer
1+274i
-/

-- # Formal Answer
noncomputable abbrev Q359.gt := fun (answer : ℂ) => answer = ((1 : ℂ) + ((274 : ℂ) * (Complex.I : ℂ) : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q359
find (answer : ℂ) s.t.
  (F : ℂ → ℂ)
  (z : ℕ → ℂ)
  (h_F : ∀ (x : ℂ), x ≠ Complex.I → F x = (x + Complex.I) / (x - Complex.I))
  (h_z : ∀ (n : ℕ), n > 0 → z n = F (z (n - 1)))
  (h_z0 : z 0 = (1 : ℂ) / 137 + Complex.I)
  (h_ne : ∀ (n : ℕ), z n ≠ Complex.I)
: (z 2002 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q359
find_all (answer : ℂ) iff
  (F : ℂ → ℂ)
  (z : ℕ → ℂ)
  (h_F : ∀ (x : ℂ), x ≠ Complex.I → F x = (x + Complex.I) / (x - Complex.I))
  (h_z : ∀ (n : ℕ), n > 0 → z n = F (z (n - 1)))
  (h_z0 : z 0 = (1 : ℂ) / 137 + Complex.I)
  (h_ne : ∀ (n : ℕ), z n ≠ Complex.I)
: (z 2002 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
