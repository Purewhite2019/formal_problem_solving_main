import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The infinite sequence $T=\{t_0,t_1,t_2,\ldots\}$ is defined as $t_0=0,$ $t_1=1,$ and $t_n=t_{n-2}+t_{n-1}$ for all integers $n>1.$ If $a,$ $b,$ $c$ are fixed non-negative integers such that \begin{align*}
a&\equiv 5\pmod {16}\\
b&\equiv 10\pmod {16}\\
c&\equiv 15\pmod {16},
\end{align*}then what is the remainder when $t_a+t_b+t_c$ is divided by $7?$

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q144.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q144
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (t : ℕ → ℕ)
  (h₀ : t 0 = 0)
  (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1))
  (h₃ : a ≡ 5 [MOD 16])
  (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16])
: (answer = (t a + t b + t c) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q144
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (t : ℕ → ℕ)
  (h₀ : t 0 = 0)
  (h₁ : t 1 = 1)
  (h₂ : ∀ n > 1, t n = t (n - 2) + t (n - 1))
  (h₃ : a ≡ 5 [MOD 16])
  (h₄ : b ≡ 10 [MOD 16])
  (h₅ : c ≡ 15 [MOD 16])
: (answer = (t a + t b + t c) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
