import FormalProblemSolving.Basic
open RingHom Set Nat

/-
## Problem
Let $X$ be the square $[0, 1] \times [0, 1]$, and let $f : X \to \mathbb{R}$ be continuous. If $\int_Y f(x, y) \, dx \, dy = 0$ for all squares $Y$ such that
\begin{itemize}
\item[(1)] $Y \subseteq X$,
\item[(2)] $Y$ has sides parallel to those of $X$,
\item[(3)] at least one of $Y$'s sides is contained in the boundary of $X$,
\end{itemize}
is it true that $f(x, y) = 0$ for all $x, y$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q49.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q49
find (answer : Prop) s.t.
  (X : Set (ℝ × ℝ))
  (hX : X = Set.prod (Icc 0 1) (Icc 0 1))
  (room : (ℝ × ℝ) → ℝ)
  (hroom : room = fun (a,b) ↦ min (min a (1 - a)) (min b (1 - b)))
: ((∀ f : (ℝ × ℝ) → ℝ, Continuous f → (∀ P ∈ X, ∫ x in (P.1 - room P)..(P.1 + room P), ∫ y in (P.2 - room P)..(P.2 + room P), f (x, y) = 0) → (∀ P ∈ X, f P = 0)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q49
find_all (answer : Prop) iff
  (X : Set (ℝ × ℝ))
  (hX : X = Set.prod (Icc 0 1) (Icc 0 1))
  (room : (ℝ × ℝ) → ℝ)
  (hroom : room = fun (a,b) ↦ min (min a (1 - a)) (min b (1 - b)))
: ((∀ f : (ℝ × ℝ) → ℝ, Continuous f → (∀ P ∈ X, ∫ x in (P.1 - room P)..(P.1 + room P), ∫ y in (P.2 - room P)..(P.2 + room P), f (x, y) = 0) → (∀ P ∈ X, f P = 0)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
