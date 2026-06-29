import FormalProblemSolving.Basic
open Nat MeasureTheory

/-
## Problem
A dart, thrown at random, hits a square target. Assuming that any two parts of the target of equal area are equally likely to be hit, find the probability that the point hit is nearer to the center than to any edge. Express your answer in the form $(a\sqrt{b}+c)/d$, where $a$, $b$, $c$, $d$ are integers and $b$, $d$ are positive.

## Answer
(4, 2, -5, 3)
-/

-- # Formal Answer
noncomputable abbrev Q126.gt := fun (answer : ℤ × ℤ × ℤ × ℤ) => answer = (((4 : ℤ), (((2 : ℤ), (((-5 : ℤ), (3 : ℤ)) : ℤ × ℤ)) : ℤ × ℤ × ℤ)) : ℤ × ℤ × ℤ × ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q126
find (answer : ℤ × ℤ × ℤ × ℤ) s.t.
  (square : Set (EuclideanSpace ℝ (Fin 2)))
  (Scloser : Set (EuclideanSpace ℝ (Fin 2)))
  (perimeter : Set (EuclideanSpace ℝ (Fin 2)))
  (center : EuclideanSpace ℝ (Fin 2))
  (square_def : square = {p | ∀ i : Fin 2, p i ∈ Set.Icc 0 1})
  (perimeter_def : perimeter = {p ∈ square | p 0 = 0 ∨ p 0 = 1 ∨ p 1 = 0 ∨ p 1 = 1})
  (center_def : center = (fun i : Fin 2 => 1 / 2))
  (hScloser : Scloser = {p ∈ square | ∀ q ∈ perimeter, dist p center < dist p q})
: (let (a, b, c, d) := answer;
    b > 0 ∧ d > 0 ∧ (¬∃ n : ℤ, n^2 = b) ∧
    (volume Scloser).toReal / (volume square).toReal = (a * Real.sqrt b + c) / d)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q126
find_all (answer : ℤ × ℤ × ℤ × ℤ) iff
  (square : Set (EuclideanSpace ℝ (Fin 2)))
  (Scloser : Set (EuclideanSpace ℝ (Fin 2)))
  (perimeter : Set (EuclideanSpace ℝ (Fin 2)))
  (center : EuclideanSpace ℝ (Fin 2))
  (square_def : square = {p | ∀ i : Fin 2, p i ∈ Set.Icc 0 1})
  (perimeter_def : perimeter = {p ∈ square | p 0 = 0 ∨ p 0 = 1 ∨ p 1 = 0 ∨ p 1 = 1})
  (center_def : center = (fun i : Fin 2 => 1 / 2))
  (hScloser : Scloser = {p ∈ square | ∀ q ∈ perimeter, dist p center < dist p q})
: (let (a, b, c, d) := answer;
    b > 0 ∧ d > 0 ∧ (¬∃ n : ℤ, n^2 = b) ∧
    (volume Scloser).toReal / (volume square).toReal = (a * Real.sqrt b + c) / d)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
