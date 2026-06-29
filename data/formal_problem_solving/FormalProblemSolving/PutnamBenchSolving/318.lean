import FormalProblemSolving.Basic
open Finset

/-
## Problem
Alice and Bob play a game in which they take turns choosing integers from $1$ to $n$. Before any integers are chosen, Bob selects a goal of 'odd' or 'even'. On the first turn, Alice chooses one of the $n$ integers. On the second turn, Bob chooses one of the remaining integers. They continue alternately choosing one of the integers that has not yet been chosen, until the $n$th turn, which is forced and ends the game. Bob wins if the parity of $\{k : \mbox{the number $k$ was chosen on the $k$th turn}\}$ matches his goal. For which values of $n$ does Bob have a winning strategy?

## Answer
the set of all positive integers
-/

-- # Formal Answer
noncomputable abbrev Q318.gt := fun (answer : Set ℕ) => answer = ({n : ℕ | (0 : ℕ) < n} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q318
find (answer : Set ℕ) s.t.
  (IsValidGame : List ℕ → Prop)
  (IsValidGame_def : ∀ g, IsValidGame g ↔ g.Nodup ∧ (∀ i ∈ g, i ∈ Icc 1 g.length))
  (parityOf : List ℕ → ZMod 2)
  (parityOf_def : ∀ g, parityOf g = ((range g.length).filter fun i ↦ g[i]! = i + 1).card)
  (ConformsToStrategy : List ℕ → (List ℕ → ℕ) → Prop)
  (ConformsToStrategy_def : ∀ g s, ConformsToStrategy g s ↔
      ∀ (i) (h : i < g.length), Odd i → g[i] = s (g.take i))
  (IsWinningFor : ℕ → (List ℕ → ℕ) → Prop)
  (IsWinningFor_def : ∀ n s, IsWinningFor n s ↔
      ∃ p, ∀ g, g.length = n → IsValidGame g → ConformsToStrategy g s → parityOf g = p)
: ({n : ℕ | 0 < n ∧ ∃ s, IsWinningFor n s} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q318
find_all (answer : Set ℕ) iff
  (IsValidGame : List ℕ → Prop)
  (IsValidGame_def : ∀ g, IsValidGame g ↔ g.Nodup ∧ (∀ i ∈ g, i ∈ Icc 1 g.length))
  (parityOf : List ℕ → ZMod 2)
  (parityOf_def : ∀ g, parityOf g = ((range g.length).filter fun i ↦ g[i]! = i + 1).card)
  (ConformsToStrategy : List ℕ → (List ℕ → ℕ) → Prop)
  (ConformsToStrategy_def : ∀ g s, ConformsToStrategy g s ↔
      ∀ (i) (h : i < g.length), Odd i → g[i] = s (g.take i))
  (IsWinningFor : ℕ → (List ℕ → ℕ) → Prop)
  (IsWinningFor_def : ∀ n s, IsWinningFor n s ↔
      ∃ p, ∀ g, g.length = n → IsValidGame g → ConformsToStrategy g s → parityOf g = p)
: ({n : ℕ | 0 < n ∧ ∃ s, IsWinningFor n s} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
