import FormalProblemSolving.Basic
open Matrix Function

/-
## Problem
Let $f: [-1, 1] \to \mathbb{R}$ be a continuous function such that
\begin{itemize}
\item[(i)]
$f(x) = \frac{2-x^2}{2} f \left( \frac{x^2}{2-x^2} \right)$ for every $x$ in $[-1, 1]$,
\item[(ii)]
$f(0) = 1$, and
\item[(iii)]
$\lim_{x \to 1^-} \frac{f(x)}{\sqrt{1-x}}$ exists and is finite.
\end{itemize}
Prove that $f$ is unique, and express $f(x)$ in closed form.

## Answer
$f(x) = \sqrt{1 - x^2}$
-/

-- # Formal Answer
noncomputable abbrev Q242.gt := fun (answer : ℝ → ℝ) => answer = fun (x : ℝ) => (√((1 : ℝ) - (x ^ (2 : ℕ) : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q242
find (answer : ℝ → ℝ) s.t.
  (S : Set ℝ)
  (hS : S = Set.Icc (-1 : ℝ) 1)
  (fsat : (ℝ → ℝ) → Prop)
  (hfsat : fsat = fun f : ℝ → ℝ => ContinuousOn f S ∧
(∀ x ∈ S, f x = ((2 - x^2)/2)*f (x^2/(2 - x^2))) ∧ f 0 = 1 ∧
(∃ y : ℝ, leftLim (fun x : ℝ => (f x)/Real.sqrt (1 - x)) 1 = y))
: (fsat answer ∧ ∀ f : ℝ → ℝ, fsat f → ∀ x ∈ S, f x = answer x)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q242
find_all (answer : ℝ → ℝ) iff
  (S : Set ℝ)
  (hS : S = Set.Icc (-1 : ℝ) 1)
  (fsat : (ℝ → ℝ) → Prop)
  (hfsat : fsat = fun f : ℝ → ℝ => ContinuousOn f S ∧
(∀ x ∈ S, f x = ((2 - x^2)/2)*f (x^2/(2 - x^2))) ∧ f 0 = 1 ∧
(∃ y : ℝ, leftLim (fun x : ℝ => (f x)/Real.sqrt (1 - x)) 1 = y))
: (fsat answer ∧ ∀ f : ℝ → ℝ, fsat f → ∀ x ∈ S, f x = answer x)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
