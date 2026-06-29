import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Tom got a Mr. Potato Head for his birthday. It came with 3 hairstyles, 2 sets of eyebrows, 1 pair of googly eyes, 2 sets of ears, and 2 sets of lips, a pair of regular shoes, and a bonus pair of clown shoes. If a complete Mr. Potato Head personality includes eyebrows, eyes, ears, lips, shoes and optionally hair, how many different wacky personalities can Tom come up with? Note that Mr. Potato Head can be bald.

Note: You cannot "mix and match".  For example, you cannot take the left eyebrow from one pair and the right eyebrow from the other pair.

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q42.gt := fun (answer : ℕ) => answer = (64 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q42
find (answer : ℕ) s.t.
  (hairstyles : ℕ)
  (eyebrows : ℕ)
  (eyes : ℕ)
  (ears : ℕ)
  (lips : ℕ)
  (regular_shoes : ℕ)
  (clown_shoes : ℕ)
  (h₀ : hairstyles = 3)
  (h₁ : eyebrows = 2)
  (h₂ : eyes = 1)
  (h₃ : ears = 2)
  (h₄ : lips = 2)
  (h₅ : regular_shoes = 1)
  (h₆ : clown_shoes = 1)
: (answer = (hairstyles + 1) * eyebrows * eyes * ears * lips * (regular_shoes + clown_shoes))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q42
find_all (answer : ℕ) iff
  (hairstyles : ℕ)
  (eyebrows : ℕ)
  (eyes : ℕ)
  (ears : ℕ)
  (lips : ℕ)
  (regular_shoes : ℕ)
  (clown_shoes : ℕ)
  (h₀ : hairstyles = 3)
  (h₁ : eyebrows = 2)
  (h₂ : eyes = 1)
  (h₃ : ears = 2)
  (h₄ : lips = 2)
  (h₅ : regular_shoes = 1)
  (h₆ : clown_shoes = 1)
: (answer = (hairstyles + 1) * eyebrows * eyes * ears * lips * (regular_shoes + clown_shoes))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
