import FormalProblemSolving.Basic
open Function Nat

/-
## Problem
Given a finite string $S$ of symbols $X$ and $O$, we write $\Delta(S)$ for the number of $X$'s in $S$ minus the number of $O$'s. For example, $\Delta(XOOXOOX)=-1$. We call a string $S$ \emph{balanced} if every substring $T$ of (consecutive symbols of) $S$ has $-2 \leq \Delta(T) \leq 2$. Thus, $XOOXOOX$ is not balanced, since it contains the substring $OOXOO$. Find, with proof, the number of balanced strings of length $n$.

## Answer
$2^{\lfloor (n + 2) / 2 \rfloor} + 2^{\lfloor (n + 1) / 2 \rfloor} - 2$
-/

-- # Formal Answer
noncomputable abbrev Q168.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) =>
  ((((2 : ℕ) ^ (⌊((n + (2 : ℕ) : ℕ) / (2 : ℕ) : ℕ)⌋₊ : ℕ) : ℕ) +
          ((2 : ℕ) ^ (⌊((n + (1 : ℕ) : ℕ) / (2 : ℕ) : ℕ)⌋₊ : ℕ) : ℕ) :
        ℕ) -
      (2 : ℕ) :
    ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q168
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (Δ : (Fin n → ℤˣ) → Fin n → Fin n → ℤ)
  (balanced : (Fin n → ℤˣ) → Prop)
  (hΔ : ∀ S, ∀ a b, a ≤ b → Δ S a b = ∑ i in Finset.Icc a b, (S i : ℤ))
  (hbalanced : ∀ S, balanced S ↔ ∀ a b, a ≤ b → |Δ S a b| ≤ 2)
: ({S : Fin n → ℤˣ | balanced S}.ncard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q168
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (Δ : (Fin n → ℤˣ) → Fin n → Fin n → ℤ)
  (balanced : (Fin n → ℤˣ) → Prop)
  (hΔ : ∀ S, ∀ a b, a ≤ b → Δ S a b = ∑ i in Finset.Icc a b, (S i : ℤ))
  (hbalanced : ∀ S, balanced S ↔ ∀ a b, a ≤ b → |Δ S a b| ≤ 2)
: ({S : Fin n → ℤˣ | balanced S}.ncard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
