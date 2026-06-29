import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For integers $n$, let \[f(n) = \left\{
\begin{array}{cl}
n^2 & \text{ if }n\text{ is odd}, \\
n^2 - 4n - 1 & \text{ if }n\text{ is even}.
\end{array}
\right.\]Find $f(f(f(f(f(4)))))$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q281.gt := fun (answer : ℤ) => answer = (1 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q281
find (answer : ℤ) s.t.
  (f : ℤ → ℤ)
  (h_f : ∀ n, f n = if n % 2 = 0 then n ^ 2 - 4 * n - 1 else n ^ 2)
: (answer = f (f (f (f (f 4)))))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q281
find_all (answer : ℤ) iff
  (f : ℤ → ℤ)
  (h_f : ∀ n, f n = if n % 2 = 0 then n ^ 2 - 4 * n - 1 else n ^ 2)
: (answer = f (f (f (f (f 4)))))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
