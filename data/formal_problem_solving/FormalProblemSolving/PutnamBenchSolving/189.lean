import FormalProblemSolving.Basic
open Nat Set Topology Filter

/-
## Problem
Fix an integer $b \geq 2$. Let $f(1) = 1$, $f(2) = 2$, and for each
$n \geq 3$, define $f(n) = n f(d)$, where $d$ is the number of
base-$b$ digits of $n$. For which values of $b$ does
\[
\sum_{n=1}^\infty \frac{1}{f(n)}
\]
converge?

## Answer
{2}
-/

-- # Formal Answer
noncomputable abbrev Q189.gt := fun (answer : Set ℕ) => answer = ({(2 : ℕ)} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q189
find (answer : Set ℕ) s.t.
  (f : ℕ → ℕ → ℝ)
  (hf : ∀ b : ℕ, f b 1 = 1 ∧ f b 2 = 2 ∧ ∀ n ∈ Ici 3, f b n = n * f b (Nat.digits b n).length)
: ({b ∈ Ici 2 | ∃ L : ℝ, Tendsto (fun m : ℕ => ∑ n in Finset.Icc 1 m, 1/(f b n)) atTop (𝓝 L)} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q189
find_all (answer : Set ℕ) iff
  (f : ℕ → ℕ → ℝ)
  (hf : ∀ b : ℕ, f b 1 = 1 ∧ f b 2 = 2 ∧ ∀ n ∈ Ici 3, f b n = n * f b (Nat.digits b n).length)
: ({b ∈ Ici 2 | ∃ L : ℝ, Tendsto (fun m : ℕ => ∑ n in Finset.Icc 1 m, 1/(f b n)) atTop (𝓝 L)} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
