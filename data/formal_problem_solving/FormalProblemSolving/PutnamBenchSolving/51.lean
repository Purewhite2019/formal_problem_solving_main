import FormalProblemSolving.Basic
open RingHom Set Nat Filter Topology

/-
## Problem
An ordered triple $(a, b, c)$ of positive irrational numbers with $a + b + c = 1$ is considered $\textit{balanced}$ if all three elements are less than $\frac{1}{2}$. If a triple is not balanced, we can perform a ``balancing act'' $B$ defined by $B(a, b, c) = (f(a), f(b), f(c))$, where $f(x) = 2x - 1$ if $x > 1/2$ and $f(x) = 2x$ otherwise. Will finitely many iterations of this balancing act always eventually produce a balanced triple?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q51.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q51
find (answer : Prop) s.t.
  (P : ℝ × ℝ × ℝ → Prop)
  (balanced : ℝ × ℝ × ℝ → Prop)
  (B : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ)
  (hP : P = fun (a, b, c) => Irrational a ∧ Irrational b ∧ Irrational c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1)
  (hbalanced : balanced = fun (a, b, c) => a < 1/2 ∧ b < 1/2 ∧ c < 1/2)
  (hB : B = fun (a, b, c) => (ite (a > 1/2) (2*a - 1) (2*a), ite (b > 1/2) (2*b - 1) (2*b), ite (c > 1/2) (2*c - 1) (2*c)))
: ((∀ t : ℝ × ℝ × ℝ, P t → ∃ n : ℕ, balanced (B^[n] t)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q51
find_all (answer : Prop) iff
  (P : ℝ × ℝ × ℝ → Prop)
  (balanced : ℝ × ℝ × ℝ → Prop)
  (B : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ)
  (hP : P = fun (a, b, c) => Irrational a ∧ Irrational b ∧ Irrational c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1)
  (hbalanced : balanced = fun (a, b, c) => a < 1/2 ∧ b < 1/2 ∧ c < 1/2)
  (hB : B = fun (a, b, c) => (ite (a > 1/2) (2*a - 1) (2*a), ite (b > 1/2) (2*b - 1) (2*b), ite (c > 1/2) (2*c - 1) (2*c)))
: ((∀ t : ℝ × ℝ × ℝ, P t → ∃ n : ℕ, balanced (B^[n] t)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
