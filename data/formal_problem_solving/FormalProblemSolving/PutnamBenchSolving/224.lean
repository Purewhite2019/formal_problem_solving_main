import FormalProblemSolving.Basic
open Topology MvPolynomial Filter Set

/-
## Problem
Let $S$ be a set of rational numbers such that
\begin{enumerate}
\item[(a)] $0 \in S$;
\item[(b)] If $x \in S$ then $x+1\in S$ and $x-1\in S$; and
\item[(c)] If $x\in S$ and $x\not\in\{0,1\}$, then $\frac{1}{x(x-1)}\in S$.
\end{enumerate}
Must $S$ contain all rational numbers?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q224.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q224
find (answer : Prop) s.t.
  (S : Set ℚ)
  (h0 : 0 ∈ S)
  (h1 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
  (h2 : ∀ x ∈ S, x ∉ ({0, 1} : Set ℚ) → 1 / (x * (x - 1)) ∈ S)
: ((∀ r : ℚ, r ∈ S) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q224
find_all (answer : Prop) iff
  (S : Set ℚ)
  (h0 : 0 ∈ S)
  (h1 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
  (h2 : ∀ x ∈ S, x ∉ ({0, 1} : Set ℚ) → 1 / (x * (x - 1)) ∈ S)
: ((∀ r : ℚ, r ∈ S) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
