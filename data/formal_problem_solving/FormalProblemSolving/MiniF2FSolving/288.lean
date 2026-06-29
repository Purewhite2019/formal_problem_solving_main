import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Which of the following is equivalent to
$(2+3)(2^2+3^2)(2^4+3^4)(2^8+3^8)(2^{16}+3^{16})(2^{32}+3^{32})(2^{64}+3^{64})?$
$\textbf{(A)} ~3^{127} + 2^{127} \qquad\textbf{(B)} ~3^{127} + 2^{127} + 2 \cdot 3^{63} + 3 \cdot 2^{63} \qquad\textbf{(C)} ~3^{128}-2^{128} \qquad\textbf{(D)} ~3^{128} + 2^{128} \qquad\textbf{(E)}

## Answer
(C) $3^{128}-2^{128}$
-/

-- # Formal Answer
noncomputable abbrev Q288.gt := fun (answer : ℤ) => answer = (((3 : ℤ) ^ (128 : ℕ) : ℤ) - ((2 : ℤ) ^ (128 : ℕ) : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q288
find (answer : ℤ) s.t.
  (product : ℤ)
  (h_product : product = ∏ k in Finset.range 7, (2^(2^k) + 3^(2^k)))
: (answer = product)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q288
find_all (answer : ℤ) iff
  (product : ℤ)
  (h_product : product = ∏ k in Finset.range 7, (2^(2^k) + 3^(2^k)))
: (answer = product)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
