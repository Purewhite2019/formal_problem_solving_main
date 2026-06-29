import FormalProblemSolving.Basic

open scoped FPS

problem quad_min.Q
  (a b c : ℝ)
find (x₀ : ℝ) s.t.
  (ha : a > 0)
  (f : ℝ → ℝ)
  (hf : f = fun x => a * x^2 + b * x + c)
: (∀ x, f x ≥ f x₀) := solve
-- case Proof
-- a b c : ℝ
-- ha : a > 0
-- f : ℝ → ℝ
-- hf : f = fun x => a * x ^ 2 + b * x + c
-- ⊢ ∀ (x : ℝ), f x ≥ f ?Answer
-- case Answer
-- a b c : ℝ
-- ⊢ ℝ
intro x
let vertex := - b / (2 * a)
have h_factorization : (a * x^2 + b * x + c) - (a * vertex^2 + b * vertex + c) = a * (x - vertex)^2 := by
    have ha_ne_zero : 2 * a ≠ 0 := by linarith
    dsimp [vertex]
    field_simp [ha_ne_zero]
    ring
rw [ge_iff_le, ← sub_nonneg]; subst hf
case Answer := - b / (2 * a)
change 0 ≤ (a * x^2 + b * x + c) - (a * vertex^2 + b * vertex + c)
rw [h_factorization]; apply mul_nonneg
· linarith
· apply sq_nonneg

noncomputable def quad_min.A := extract_answer quad_min.Q
#print quad_min.A
-- def quad_min.A : ℝ → ℝ → ℝ → ℝ :=
-- fun a b c => -b / (2 * a)

noncomputable abbrev quad_min.gt := fun (a b c : ℝ) => - b / 2 / a

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
