import Lean
import Mathlib

open Lean Elab Command Term Meta Tactic

set_option autoImplicit false
set_option relaxedAutoImplicit false

structure ProblemSol {α : Type} (P : α → Prop) where
  Answer : α
  Proof : P Answer

def ProblemTarget (p : Prop) : Prop := p

def getRawBody (e : Expr) (i : Nat := 2) : MetaM Expr := do
  lambdaTelescope e fun args body => do
    let body ← whnf body
    if body.isAppOf ``ProblemSol.mk then
      let val := body.getAppArgs[i]!
      mkLambdaFVars args val
    else
      throwError "Extraction failed: Body is not a ProblemSol constructor. Body: {body}"

elab "extract_ans_type " id:ident : term => do
  let name ← resolveGlobalConstNoOverload id
  let info ← getConstInfoDefn name
  getRawBody info.value 0

elab "extract_predicate " id:ident : term => do
  let name ← resolveGlobalConstNoOverload id
  let info ← getConstInfoDefn name
  getRawBody info.value 1

elab "extract_answer " id:ident : term => do
  let name ← resolveGlobalConstNoOverload id
  let info ← getConstInfoDefn name
  getRawBody info.value 2

elab "extract_proof " id:ident : term => do
  let name ← resolveGlobalConstNoOverload id
  let info ← getConstInfoDefn name
  getRawBody info.value 3


namespace FPS
scoped syntax (name := problemSyntax)
  "problem" ident bracketedBinder* "find" "(" ident ":" term ")"
  "s.t." bracketedBinder* ":" term ":=" term : command

scoped macro_rules
| `(problem $name:ident $vars* find ($ansId:ident : $ansType:term) s.t. $hyps* : $concl:term := $body:term) => do
  let proofSpec ← `(fun $ansId:ident : $ansType => ∀ $hyps*, ProblemTarget $concl)
  `(@[reducible] noncomputable def $name $vars* : ProblemSol $proofSpec := $body)

scoped syntax (name := solveSyntax) "solve" (ppSpace tacticSeq)? : term

@[term_elab solveSyntax]
def elabSolve : TermElab := fun stx expectedType? => do
  let some expectedType := expectedType?
    | throwError "`solve` must be used in a position of known type"

  let mvar ← mkFreshExprMVar expectedType
  let mvarId := mvar.mvarId!
  let setupTx ← `(tactic|
    apply ProblemSol.mk;
    intros;
    unfold ProblemTarget
  )

  let _ ← Lean.Elab.Tactic.run mvarId do
    withOptions (fun o => o.setBool `tactic.hygienic false) do
      evalTactic setupTx
    withTacticInfoContext stx[0] (pure ())
    if let some seq := stx[1].getOptional? then
      evalTactic seq

  return mvar

end FPS


namespace DFPS

scoped syntax (name := problemSyntax)
  "problem" ident bracketedBinder* "find_all" "(" ident ":" term ")"
  "iff" bracketedBinder* ":" term ":=" term : command

scoped macro_rules
| `(problem $name:ident $vars* find_all ($ansId:ident : $ansType:term) iff $hyps* : $concl:term := $body:term) => do
  let proofSpec ← `(fun answer_predicate :  Prop => ∀ $hyps*, (ProblemTarget $concl) ↔ (answer_predicate))
  `(@[reducible] noncomputable def $name $vars* ($ansId : $ansType) : ProblemSol $proofSpec := $body)

scoped syntax (name := solveSyntax) "solve" (ppSpace tacticSeq)? : term
@[term_elab solveSyntax]
def elabSolve : TermElab := fun stx expectedType? => do
  let some expectedType := expectedType?
    | throwError "`solve` must be used in a position of known type"

  let hAnswerName := mkIdent `_h_answer
  let hConclusionName := mkIdent `_h_conclusion
  let mvar ← mkFreshExprMVar expectedType
  let mvarId := mvar.mvarId!

  let setupTx ← `(tactic|
    apply ProblemSol.mk;
    intros;
    unfold ProblemTarget;
    refine @Iff.intro ?_ _ ?Forward ?Backward;
    rotate_left;
    intros $hAnswerName;
    rotate_right;
    intros $hConclusionName;
  )

  let _ ← Lean.Elab.Tactic.run mvarId do
    withOptions (fun o => o.setBool `tactic.hygienic false) do
      evalTactic setupTx
    withTacticInfoContext stx[0] (pure ())
    if let some seq := stx[1].getOptional? then
      evalTactic seq

  return mvar

end DFPS
