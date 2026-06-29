import FormalProblemSolving.Basic
open Topology Filter Set Polynomial Function

/-
## Problem
Let $B(n)$ be the number of ones in the base two expression for the positive integer $n$. For example, $B(6)=B(110_2)=2$ and $B(15)=B(1111_2)=4$. Determine whether or not $\exp \left(\sum_{n=1}^\infty \frac{B(n)}{n(n+1)}\right)$ is a rational number. Here $\exp(x)$ denotes $e^x$.

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q73.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q73
find (answer : Prop) s.t.
  (sumbits : List ℕ → ℤ)
  (B : ℕ → ℤ)
  (hsumbits : ∀ bits : List ℕ, sumbits bits = ∑ i : Fin bits.length, (bits[i] : ℤ))
  (hB : ∀ n > 0, B n = sumbits (Nat.digits 2 n))
: (answer ↔ (∃ q : ℚ, Real.exp (∑' n : Set.Ici 1, B n / ((n : ℝ) * ((n : ℝ) + 1))) = q))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q73
find_all (answer : Prop) iff
  (sumbits : List ℕ → ℤ)
  (B : ℕ → ℤ)
  (hsumbits : ∀ bits : List ℕ, sumbits bits = ∑ i : Fin bits.length, (bits[i] : ℤ))
  (hB : ∀ n > 0, B n = sumbits (Nat.digits 2 n))
: (answer ↔ (∃ q : ℚ, Real.exp (∑' n : Set.Ici 1, B n / ((n : ℝ) * ((n : ℝ) + 1))) = q))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
