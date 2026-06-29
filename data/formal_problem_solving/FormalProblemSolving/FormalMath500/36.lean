import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Karla drove her vehicle 12,000 miles last year. How many gallons of gasoline would she have saved last year had she driven a fuel-efficient hybrid car averaging 48 miles per gallon rather than an SUV averaging 15 miles per gallon?

## Answer
550
-/

-- # Formal Answer
noncomputable abbrev Q36.gt := fun (answer : ℕ) => answer = (550 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q36
find (answer : ℕ) s.t.
  (miles_driven : ℕ)
  (hybrid_mpg : ℕ)
  (suv_mpg : ℕ)
  (gallons_hybrid : ℕ)
  (gallons_suv : ℕ)
  (h_miles_driven : miles_driven = 12000)
  (h_hybrid_mpg : 48 = hybrid_mpg)
  (h_suv_mpg : 15 = suv_mpg)
  (h_gallons_hybrid : miles_driven / hybrid_mpg = gallons_hybrid)
  (h_gallons_suv : miles_driven / suv_mpg = gallons_suv)
: (gallons_suv - gallons_hybrid = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q36
find_all (answer : ℕ) iff
  (miles_driven : ℕ)
  (hybrid_mpg : ℕ)
  (suv_mpg : ℕ)
  (gallons_hybrid : ℕ)
  (gallons_suv : ℕ)
  (h_miles_driven : miles_driven = 12000)
  (h_hybrid_mpg : 48 = hybrid_mpg)
  (h_suv_mpg : 15 = suv_mpg)
  (h_gallons_hybrid : miles_driven / hybrid_mpg = gallons_hybrid)
  (h_gallons_suv : miles_driven / suv_mpg = gallons_suv)
: (gallons_suv - gallons_hybrid = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
