import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
One ticket to a show costs $\$20$ at full price. Susan buys $4$ tickets using a coupon that gives her a $25\%$ discount. Pam buys $5$ tickets using a coupon that gives her a $30\%$ discount. How many more dollars does Pam pay than Susan?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q213.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q213
find (answer : ℝ) s.t.
  (full_price : ℝ)
  (susan_tickets : ℕ)
  (pam_tickets : ℕ)
  (susan_discount : ℝ)
  (pam_discount : ℝ)
  (h₀ : full_price = 20)
  (h₁ : susan_tickets = 4)
  (h₂ : susan_discount = 0.25)
  (h₃ : pam_tickets = 5)
  (h₄ : pam_discount = 0.30)
: ((pam_tickets * full_price * (1 - pam_discount)) - (susan_tickets * full_price * (1 - susan_discount)) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q213
find_all (answer : ℝ) iff
  (full_price : ℝ)
  (susan_tickets : ℕ)
  (pam_tickets : ℕ)
  (susan_discount : ℝ)
  (pam_discount : ℝ)
  (h₀ : full_price = 20)
  (h₁ : susan_tickets = 4)
  (h₂ : susan_discount = 0.25)
  (h₃ : pam_tickets = 5)
  (h₄ : pam_discount = 0.30)
: ((pam_tickets * full_price * (1 - pam_discount)) - (susan_tickets * full_price * (1 - susan_discount)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
