import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Recently, Frank took a one-hundred question aptitude test where each correct answer scored $5$ points, each incorrect answer scored $-2$ points, and each question not answered scored zero points. Frank answered $80$ questions and scored $232$ points. How many questions did he answer correctly?

## Answer
56
-/

-- # Formal Answer
noncomputable abbrev Q179.gt := fun (answer : ℕ) => answer = (56 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q179
find (answer : ℕ) s.t.
  (total_questions : ℕ)
  (correct : ℕ)
  (incorrect : ℕ)
  (unanswered : ℕ)
  (h₀ : total_questions = 100)
  (h₁ : correct + incorrect + unanswered = total_questions)
  (h₂ : correct + incorrect = 80)
  (h₃ : 5 * correct - 2 * incorrect = 232)
: correct = answer
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q179
find_all (answer : ℕ) iff
  (total_questions : ℕ)
  (correct : ℕ)
  (incorrect : ℕ)
  (unanswered : ℕ)
  (h₀ : total_questions = 100)
  (h₁ : correct + incorrect + unanswered = total_questions)
  (h₂ : correct + incorrect = 80)
  (h₃ : 5 * correct - 2 * incorrect = 232)
: correct = answer
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
