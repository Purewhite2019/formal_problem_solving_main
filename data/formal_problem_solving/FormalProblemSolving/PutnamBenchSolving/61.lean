import FormalProblemSolving.Basic

/-
## Problem
Let $r$ and $s$ be positive integers. Derive a formula for the number of ordered quadruples $(a,b,c,d)$ of positive integers such that $3^r \cdot 7^s=\text{lcm}[a,b,c]=\text{lcm}[a,b,d]=\text{lcm}[a,c,d]=\text{lcm}[b,c,d]$. The answer should be a function of $r$ and $s$. (Note that $\text{lcm}[x,y,z]$ denotes the least common multiple of $x,y,z$.)

## Answer
(fun r s : ℕ => (1 + 4 * r + 6 * r ^ 2) * (1 + 4 * s + 6 * s ^ 2))
-/

-- # Formal Answer
noncomputable abbrev Q61.gt := fun (r : ℕ) (s : ℕ) (answer : ℕ → ℕ → ℕ) => answer = fun (r s : ℕ) =>
  ((((1 : ℕ) + ((4 : ℕ) * r : ℕ) : ℕ) + ((6 : ℕ) * (r ^ (2 : ℕ) : ℕ) : ℕ) : ℕ) *
      (((1 : ℕ) + ((4 : ℕ) * s : ℕ) : ℕ) + ((6 : ℕ) * (s ^ (2 : ℕ) : ℕ) : ℕ) : ℕ) :
    ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q61
  (r : ℕ)
  (s : ℕ)
find (answer : ℕ → ℕ → ℕ) s.t.
  (abcdlcm : ℕ → ℕ → ℕ → ℕ → Prop)
  (rspos : r > 0 ∧ s > 0)
  (habcdlcm : ∀ a b c d : ℕ, abcdlcm a b c d ↔
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a b) c) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a b) d) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a c) d) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm b c) d)))
: ({h : ℕ × ℕ × ℕ × ℕ | abcdlcm h.1 h.2.1 h.2.2.1 h.2.2.2}.encard = answer r s)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q61
  (r : ℕ)
  (s : ℕ)
find_all (answer : ℕ → ℕ → ℕ) iff
  (abcdlcm : ℕ → ℕ → ℕ → ℕ → Prop)
  (rspos : r > 0 ∧ s > 0)
  (habcdlcm : ∀ a b c d : ℕ, abcdlcm a b c d ↔
      (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a b) c) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a b) d) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm a c) d) ∧
      (3 ^ r * 7 ^ s = Nat.lcm (Nat.lcm b c) d)))
: ({h : ℕ × ℕ × ℕ × ℕ | abcdlcm h.1 h.2.1 h.2.2.1 h.2.2.2}.encard = answer r s)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
