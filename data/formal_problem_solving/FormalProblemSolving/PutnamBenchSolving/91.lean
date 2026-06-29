import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Prove or disprove the following statement: If $F$ is a finite set with two or more elements, then there exists a binary operation $*$ on F such that for all $x,y,z$ in $F$,
\begin{enumerate}
\item[(i)] $x*z=y*z$ implies $x=y$ (right cancellation holds), and
\item[(ii)] $x*(y*z) \neq (x*y)*z$ (\emph{no} case of associativity holds).
\end{enumerate}

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q91.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q91
find (answer : Prop) s.t.
: ((∀ (F : Type*) (_ : Fintype F), Fintype.card F ≥ 2 → (∃ mul : F → F → F, ∀ x y z : F, (mul x z = mul y z → x = y) ∧ (mul x (mul y z) ≠ mul (mul x y) z))) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q91
find_all (answer : Prop) iff
: ((∀ (F : Type*) (_ : Fintype F), Fintype.card F ≥ 2 → (∃ mul : F → F → F, ∀ x y z : F, (mul x z = mul y z → x = y) ∧ (mul x (mul y z) ≠ mul (mul x y) z))) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
