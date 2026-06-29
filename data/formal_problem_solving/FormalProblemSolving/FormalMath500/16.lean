import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The data in the table below shows the percent of bus riders in a survey of Central H.S. students; 300 males and 300 females in each grade were surveyed. For which grade is the number of its male bus riders closest to $135\%$ of the number of its female bus riders? \begin{tabular}{|l|c|c|}\hline
Grade & Males & Females \\ \hline
$9^{\mathrm{th}}$ grade & $41.1\%$ & $39.4\%$ \\ \hline
$10^{\mathrm{th}}$ grade & $34.4\%$ & $33.1\%$ \\ \hline
$11^{\mathrm{th}}$ grade & $20.6\%$ & $13.8\%$ \\ \hline
$12^{\mathrm{th}}$ grade & $11.6\%$ & $8.6\%$ \\ \hline
\end{tabular}

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q16.gt := fun (answer : ℕ) => answer = (12 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q16
find (answer : ℕ) s.t.
  (grade : ℕ)
  (males : ℕ → ℝ)
  (females : ℕ → ℝ)
  (h_males_9th : males 9 = 0.411 * 300)
  (h_females_9th : females 9 = 0.394 * 300)
  (h_males_10th : males 10 = 0.344 * 300)
  (h_females_10th : females 10 = 0.331 * 300)
  (h_males_11th : males 11 = 0.206 * 300)
  (h_females_11th : females 11 = 0.138 * 300)
  (h_males_12th : males 12 = 0.116 * 300)
  (h_females_12th : females 12 = 0.086 * 300)
: (answer ∈ [9, 10, 11, 12] ∧  (∀ (g : ℕ), g ∈ [9, 10, 11, 12] → abs (males g - 1.35 * females g) ≥ abs (males answer - 1.35 * females answer)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q16
find_all (answer : ℕ) iff
  (grade : ℕ)
  (males : ℕ → ℝ)
  (females : ℕ → ℝ)
  (h_males_9th : males 9 = 0.411 * 300)
  (h_females_9th : females 9 = 0.394 * 300)
  (h_males_10th : males 10 = 0.344 * 300)
  (h_females_10th : females 10 = 0.331 * 300)
  (h_males_11th : males 11 = 0.206 * 300)
  (h_females_11th : females 11 = 0.138 * 300)
  (h_males_12th : males 12 = 0.116 * 300)
  (h_females_12th : females 12 = 0.086 * 300)
: (answer ∈ [9, 10, 11, 12] ∧  (∀ (g : ℕ), g ∈ [9, 10, 11, 12] → abs (males g - 1.35 * females g) ≥ abs (males answer - 1.35 * females answer)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
