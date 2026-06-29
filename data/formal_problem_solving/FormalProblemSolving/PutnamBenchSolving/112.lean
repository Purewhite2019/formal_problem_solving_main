import FormalProblemSolving.Basic

/-
## Problem
Let $\vec{G}(x,y)=\left(\frac{-y}{x^2+4y^2},\frac{x}{x^2+4y^2},0\right)$. Prove or disprove that there is a vector-valued function $\vec{F}(x,y,z)=(M(x,y,z),N(x,y,z),P(x,y,z))$ with the following properties:
\begin{enumerate}
\item[(i)] $M$, $N$, $P$ have continuous partial derivatives for all $(x,y,z) \neq (0,0,0)$;
\item[(ii)] $\text{Curl}\,\vec{F}=\vec{0}$ for all $(x,y,z) \neq (0,0,0)$;
\item[(iii)] $\vec{F}(x,y,0)=\vec{G}(x,y)$.
\end{enumerate}

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q112.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q112
find (answer : Prop) s.t.
  (vec2 : ℝ → ℝ → (Fin 2 → ℝ))
  (vec3 : ℝ → ℝ → ℝ → (Fin 3 → ℝ))
  (G : (Fin 2 → ℝ) → (Fin 3 → ℝ))
  (hG : G = (fun v : Fin 2 → ℝ => vec3 (-v 1 / ((v 0) ^ 2 + 4 * (v 1) ^ 2)) (v 0 / ((v 0) ^ 2 + 4 * (v 1) ^ 2)) 0))
  (vrepl : (Fin 3 → ℝ) → Fin 3 → ℝ → (Fin 3 → ℝ))
  (hvrepl : vrepl = (fun (v : Fin 3 → ℝ) (i : Fin 3) (vi : ℝ) => (fun j : Fin 3 => if j = i then vi else v j)))
  (contdiffv : ((Fin 3 → ℝ) → ℝ) → Fin 3 → (Fin 3 → ℝ) → Prop)
  (hcontdiffv : contdiffv = (fun (Fi : (Fin 3 → ℝ) → ℝ) (j : Fin 3) (v : Fin 3 → ℝ) => ContDiffAt ℝ 1 (fun vj : ℝ => Fi (vrepl v j vj)) (v j)))
  (partderiv : ((Fin 3 → ℝ) → ℝ) → Fin 3 → ((Fin 3 → ℝ) → ℝ))
  (hpartderiv : partderiv = (fun (Fi : (Fin 3 → ℝ) → ℝ) (j : Fin 3) => (fun v : Fin 3 → ℝ => deriv (fun vj : ℝ => Fi (vrepl v j vj)) (v j))))
  (Fprop1 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (Fprop2 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (Fprop3 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (hFprop1 : Fprop1 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ i : Fin 3, ∀ j : Fin 3, ∀ v ≠ 0, contdiffv (F i) j v))
  (hFprop2 : Fprop2 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ v ≠ 0, vec3 ((partderiv (F 2) 1 - partderiv (F 1) 2) v) ((partderiv (F 0) 2 - partderiv (F 2) 0) v) ((partderiv (F 1) 0 - partderiv (F 0) 1) v) = 0))
  (hFprop3 : Fprop3 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ x y : ℝ, (fun i : Fin 3 => (F i) (vec3 x y 0)) = G (vec2 x y)))
  (hvec2 : ∀ x y : ℝ, (vec2 x y) 0 = x ∧ (vec2 x y) 1 = y)
  (hvec3 : ∀ x y z : ℝ, (vec3 x y z) 0 = x ∧ (vec3 x y z) 1 = y ∧ (vec3 x y z) 2 = z)
: ((∃ F : Fin 3 → ((Fin 3 → ℝ) → ℝ), Fprop1 F ∧ Fprop2 F ∧ Fprop3 F) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q112
find_all (answer : Prop) iff
  (vec2 : ℝ → ℝ → (Fin 2 → ℝ))
  (vec3 : ℝ → ℝ → ℝ → (Fin 3 → ℝ))
  (G : (Fin 2 → ℝ) → (Fin 3 → ℝ))
  (hG : G = (fun v : Fin 2 → ℝ => vec3 (-v 1 / ((v 0) ^ 2 + 4 * (v 1) ^ 2)) (v 0 / ((v 0) ^ 2 + 4 * (v 1) ^ 2)) 0))
  (vrepl : (Fin 3 → ℝ) → Fin 3 → ℝ → (Fin 3 → ℝ))
  (hvrepl : vrepl = (fun (v : Fin 3 → ℝ) (i : Fin 3) (vi : ℝ) => (fun j : Fin 3 => if j = i then vi else v j)))
  (contdiffv : ((Fin 3 → ℝ) → ℝ) → Fin 3 → (Fin 3 → ℝ) → Prop)
  (hcontdiffv : contdiffv = (fun (Fi : (Fin 3 → ℝ) → ℝ) (j : Fin 3) (v : Fin 3 → ℝ) => ContDiffAt ℝ 1 (fun vj : ℝ => Fi (vrepl v j vj)) (v j)))
  (partderiv : ((Fin 3 → ℝ) → ℝ) → Fin 3 → ((Fin 3 → ℝ) → ℝ))
  (hpartderiv : partderiv = (fun (Fi : (Fin 3 → ℝ) → ℝ) (j : Fin 3) => (fun v : Fin 3 → ℝ => deriv (fun vj : ℝ => Fi (vrepl v j vj)) (v j))))
  (Fprop1 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (Fprop2 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (Fprop3 : (Fin 3 → ((Fin 3 → ℝ) → ℝ)) → Prop)
  (hFprop1 : Fprop1 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ i : Fin 3, ∀ j : Fin 3, ∀ v ≠ 0, contdiffv (F i) j v))
  (hFprop2 : Fprop2 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ v ≠ 0, vec3 ((partderiv (F 2) 1 - partderiv (F 1) 2) v) ((partderiv (F 0) 2 - partderiv (F 2) 0) v) ((partderiv (F 1) 0 - partderiv (F 0) 1) v) = 0))
  (hFprop3 : Fprop3 = (fun F : Fin 3 → ((Fin 3 → ℝ) → ℝ) => ∀ x y : ℝ, (fun i : Fin 3 => (F i) (vec3 x y 0)) = G (vec2 x y)))
  (hvec2 : ∀ x y : ℝ, (vec2 x y) 0 = x ∧ (vec2 x y) 1 = y)
  (hvec3 : ∀ x y z : ℝ, (vec3 x y z) 0 = x ∧ (vec3 x y z) 1 = y ∧ (vec3 x y z) 2 = z)
: ((∃ F : Fin 3 → ((Fin 3 → ℝ) → ℝ), Fprop1 F ∧ Fprop2 F ∧ Fprop3 F) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
