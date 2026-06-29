import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Linda, Sherry, June, and Connie walked around their neighborhoods selling girl scout cookies. Linda earned $\$27.47$, Sherry earned $\$35.23$, June earned $\$37.37$, and Connie earned $\$26.52$. After the sale, they pooled their money together and went to the bank to convert the coins into dollars. How much money, in cents, was left after they converted as many of the coins into bills as they could?

## Answer
59
-/

-- # Formal Answer
noncomputable abbrev Q236.gt := fun (answer : ℕ) => answer = (59 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q236
find (answer : ℕ) s.t.
  (h_total_cents : ℕ)
  (ht : h_total_cents = (2747 + 3523 + 3737 + 2652))
: (answer = h_total_cents % 100)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q236
find_all (answer : ℕ) iff
  (h_total_cents : ℕ)
  (ht : h_total_cents = (2747 + 3523 + 3737 + 2652))
: (answer = h_total_cents % 100)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
