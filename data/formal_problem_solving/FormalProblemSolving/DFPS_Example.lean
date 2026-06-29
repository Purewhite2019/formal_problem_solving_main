import FormalProblemSolving.Basic

open scoped DFPS

problem quad_min.Q
  (a b c : ℝ)
find_all (x₀ : ℝ) iff
  (ha : a > 0)
  (f : ℝ → ℝ)
  (hf : f = fun x => a * x^2 + b * x + c)
: (∀ x, f x ≥ f x₀) := solve
-- case Forward
-- a b c x₀ : ℝ
-- ha : a > 0
-- f : ℝ → ℝ
-- hf : f = fun x => a * x ^ 2 + b * x + c
-- _h_conclusion : ∀ (x : ℝ), f x ≥ f x₀
-- ⊢ ?Answer
-- case Backward
-- a b c x₀ : ℝ
-- ha : a > 0
-- f : ℝ → ℝ
-- hf : f = fun x => a * x ^ 2 + b * x + c
-- _h_answer : ?Answer
-- ⊢ ∀ (x : ℝ), f x ≥ f x₀
-- case Answer
-- a b c x₀ : ℝ
-- ⊢ Prop

-- We rewrite \(f(x)\) in vertex form:
-- \(f(x) = a\left(x - v\right)^2 + \left(c - \frac{b^2}{4a}\right)\), where \(v = -\frac{b}{2a}\).

-- From the hypothesis, \(f(v) \geq f(x_0)\).
-- Substituting the vertex form into this inequality yields
-- \(a (x_0 - v)^2 \leq 0\).

-- Because \(a > 0\) and a square is always nonnegative, we must have \((x_0 - v)^2 = 0\).
-- Thus \(x_0 = v = -\frac{b}{2a}\).

-- Therefore the unique minimizer is \(x_0 = -\frac{b}{2a}\).
case Forward =>
  let v := -b / (2 * a)
  have h_vertex_form : ∀ x, f x = a * (x - v)^2 + (c - b^2 / (4 * a)) := by
    aesop; field_simp [ne_of_gt ha]; ring
  have h_fv_val : f v = c - b^2 / (4 * a) := by
    rw [h_vertex_form v] ;simp
  have h_ineq : f v ≥ f x₀ := _h_conclusion v
  rw [h_fv_val, h_vertex_form x₀] at h_ineq
  have h_nonpos : a * (x₀ - v)^2 ≤ 0 := by
    linarith [h_ineq]
  have h_sq_zero : (x₀ - v)^2 = 0 := by
    nlinarith [ha, h_nonpos, sq_nonneg (x₀ - v)]
  have h_diff_zero : x₀ - v = 0 := pow_eq_zero h_sq_zero
  have h_answer : x₀ = -b / (2 * a) := eq_of_sub_eq_zero h_diff_zero
  exact h_answer

case Backward =>
  subst _h_answer
  intro x
  rw [hf]; simp;
  apply le_of_sub_nonneg
  have h_quad : a * x ^ 2 + b * x - (a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a))) =
              a * (x + b / (2 * a)) ^ 2 := by field_simp; ring
  rw [h_quad]
  positivity

noncomputable def quad_min.A := extract_answer quad_min.Q
#print quad_min.A
-- def quad_min.A : ℝ → ℝ → ℝ → ℝ → Prop :=
-- fun a b c x₀ => x₀ = -b / (2 * a)

noncomputable abbrev quad_min.gt := fun (a b c x₀ : ℝ) => x₀ = - b / 2 / a

lemma quat_min.is_correct : extract_answer quad_min.Q = quad_min.gt := by
  unfold quad_min.gt
  try ext
  try funext
  repeat {
    try norm_num;
    try ring_nf;
    try rfl;
    try aesop;
    try rw_search;
  }
