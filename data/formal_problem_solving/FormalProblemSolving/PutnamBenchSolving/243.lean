import FormalProblemSolving.Basic
open Matrix Function

/-
## Problem
Let $\FF_p$ denote the field of integers modulo a prime $p$, and let $n$ be a positive integer. Let $v$ be a fixed vector in $\FF_p^n$, let $M$ be an $n \times n$ matrix with entries of $\FF_p$, and define $G: \FF_p^n \to \FF_p^n$ by $G(x) = v + Mx$. Let $G^{(k)}$ denote the $k$-fold composition of $G$ with itself, that is, $G^{(1)}(x) = G(x)$ and $G^{(k+1)}(x) = G(G^{(k)}(x))$. Determine all pairs $p, n$ for which there exist $v$ and $M$ such that the $p^n$ vectors $G^{(k)}(0)$, $k=1,2,\dots,p^n$ are distinct.

## Answer
{q | let ⟨n, _⟩ := q; n = 1} ∪ {(2,2)}
-/

-- # Formal Answer
noncomputable abbrev Q243.gt := fun (n : ℕ) (answer : Set (ℕ × ℕ)) => answer =
  (({((n, snd) : ℕ × ℕ) : ℕ × ℕ | n = (1 : ℕ)} : Set (ℕ × ℕ)) ∪ ({(((2 : ℕ), (2 : ℕ)) : ℕ × ℕ)} : Set (ℕ × ℕ)) :
    Set (ℕ × ℕ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q243
  (n : ℕ)
find (answer : Set (ℕ × ℕ)) s.t.
  (p : ℕ)
  (hn : n > 0)
  (hp : Nat.Prime p)
  (F : Type*)
  (_ : Field F)
  (_ : Fintype F)
  (hK : Fintype.card F = p)
  (G : Matrix (Fin n) (Fin n) F → (Fin n → F) → (Fin n → F) → (Fin n → F))
  (hG : ∀ M v x, G M v x = v + mulVec M x)
: ((n, p) ∈ answer ↔
    ∃ᵉ (M : Matrix (Fin n) (Fin n) F) (v : (Fin n → F)),
      ¬(∃ i j : Finset.range (p^n), i ≠ j ∧ (G M v)^[i + 1] 0 = (G M v)^[j + 1] 0))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q243
  (n : ℕ)
find_all (answer : Set (ℕ × ℕ)) iff
  (p : ℕ)
  (hn : n > 0)
  (hp : Nat.Prime p)
  (F : Type*)
  (_ : Field F)
  (_ : Fintype F)
  (hK : Fintype.card F = p)
  (G : Matrix (Fin n) (Fin n) F → (Fin n → F) → (Fin n → F) → (Fin n → F))
  (hG : ∀ M v x, G M v x = v + mulVec M x)
: ((n, p) ∈ answer ↔
    ∃ᵉ (M : Matrix (Fin n) (Fin n) F) (v : (Fin n → F)),
      ¬(∃ i j : Finset.range (p^n), i ≠ j ∧ (G M v)^[i + 1] 0 = (G M v)^[j + 1] 0))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
