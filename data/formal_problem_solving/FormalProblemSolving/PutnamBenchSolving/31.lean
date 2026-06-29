import FormalProblemSolving.Basic
open Set Nat Polynomial

/-
## Problem
Given $n$, let $k(n)$ be the minimal degree of any monic integral polynomial $f$ such that the value of $f(x)$ is divisible by $n$ for every integer $x$. Find the value of $k(1000000)$.

## Answer
25
-/

-- # Formal Answer
noncomputable abbrev Q31.gt := fun (answer : ℕ) => answer = (25 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q31
find (answer : ℕ) s.t.
  (hdivnallx : Polynomial ℤ → Prop)
  (hdivnallx_def : hdivnallx = fun f => Monic f ∧ (∀ x : ℤ, (10^6 : ℤ) ∣ f.eval x))
: (sInf {d : ℕ | ∃ f : Polynomial ℤ, hdivnallx f ∧ d = f.natDegree} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q31
find_all (answer : ℕ) iff
  (hdivnallx : Polynomial ℤ → Prop)
  (hdivnallx_def : hdivnallx = fun f => Monic f ∧ (∀ x : ℤ, (10^6 : ℤ) ∣ f.eval x))
: (sInf {d : ℕ | ∃ f : Polynomial ℤ, hdivnallx f ∧ d = f.natDegree} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
