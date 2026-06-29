import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The function $f(n)$ is defined on the positive integers and takes non-negative integer values. $f(2)=0,f(3)>0,f(9999)=3333$ and for all $m,n:$$ f(m+n)-f(m)-f(n)=0 \text{ or } 1. $Determine $f(1982)$.

## Answer
660
-/

-- # Formal Answer
noncomputable abbrev Q230.gt := fun (answer : ℕ) => answer = (660 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q230
find (answer : ℕ) s.t.
  (f : ℕ → ℕ)
  (h₀ : ∀ (m n : ℕ), 0 < m → 0 < n →
        (f (m + n) : ℤ) - (f m : ℤ) - (f n : ℤ) = 0 ∨ (f (m + n) : ℤ) - (f m : ℤ) - (f n : ℤ) = 1)
  (h₁ : f 2 = 0)
  (h₂ : 0 < f 3)
  (h₃ : f 9999 = 3333)
: (answer = f 1982)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q230
find_all (answer : ℕ) iff
  (f : ℕ → ℕ)
  (h₀ : ∀ (m n : ℕ), 0 < m → 0 < n →
        (f (m + n) : ℤ) - (f m : ℤ) - (f n : ℤ) = 0 ∨ (f (m + n) : ℤ) - (f m : ℤ) - (f n : ℤ) = 1)
  (h₁ : f 2 = 0)
  (h₂ : 0 < f 3)
  (h₃ : f 9999 = 3333)
: (answer = f 1982)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
