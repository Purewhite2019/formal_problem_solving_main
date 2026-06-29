from typing import Optional, List, Tuple

from common.constants import CORE_OPTIONS, OPEN_HEADER, RPE_TACTICS, H_SUBMISSION_NAME, H_CONCLUSION_NAME
from common.utils import (
    format_variable_sequence,
    to_sync)
from common.pantograph.server import Server
from common.pantograph.dataclasses import GoalState, FormalProblem


class BaseSolvingServer:
    def __init__(
        self,
        imports: List[str]=["Mathlib", "Aesop"],
        project_path: Optional[str]=None,
        timeout: int=300,
        tag: str=''
    ) -> None:
        self.sample : Optional[FormalProblem] = None
        self.answer_mvarId : Optional[str] = None

        self.server: Optional[Server] = None
        self.imports = imports
        self.project_path = project_path
        self.timeout = timeout
        self.tag = tag

    async def init_backward_state_async(self, sample: Optional[FormalProblem]=None) -> GoalState:
        await self.load_problem_async(sample)

        self.answer_mvarId = None
        
        
        V = [f"  ({v.name or '_'} : {v.t})" for v in sample.independent_variables]
        Phi = [f"  ({v.name or '_'} : {v.t})" for v in sample.hypotheses]
        Psi = sample.conclusions

        self.formal_problem_framework = 'example\n' + \
            '\n'.join(
                V + \
                [f'  (answer : {self.sample.formal_answer_type})', f'  ({H_SUBMISSION_NAME}: {self.sample.formal_answer})'] + \
                Phi
            ) + \
            '\n: ' + ' ∧ '.join(Psi) + '\n:= by\n' + \
            'all_goals sorry'

        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        return units[-1].goal_state
            
    init_backward_state = to_sync(init_backward_state_async)

    async def load_problem_async(self, sample: Optional[FormalProblem]) -> None:
        if sample is None:
            return None
        
        assert isinstance(sample, FormalProblem)
        self.sample = sample
        self.server = await Server.create(
            imports=self.imports,
            project_path=self.project_path,
            core_options=CORE_OPTIONS,
            timeout=self.timeout,
        )
        return None
    
    load_problem = to_sync(load_problem_async)

    async def get_submission_async(self, state: GoalState) -> Optional[str]:
        assert self.answer_mvarId is not None
        rs = await self.server.goal_print_async(state, False, False, False, [self.answer_mvarId])
        answer = rs['extraMVars'][0]
        return answer.get('pp', None)

    get_submission = to_sync(get_submission_async)

# Answer as Term
class TermSolvingServer(BaseSolvingServer):
    async def init_solving_state_async(self, sample: Optional[FormalProblem]=None) -> Tuple[GoalState, List[str]]:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        V = [f"  ({v.name or '_'} : {v.t})" for v in sample.independent_variables]
        Phi = [f"  ({v.name or '_'} : {v.t})" for v in sample.hypotheses]
        Psi = sample.conclusions

        init_steps = ([
            'intros ' + ' '.join([v.name or '_' for v in self.sample.independent_variables])
                ] if len(self.sample.independent_variables) > 0 else []) + [
            'apply Exists.intro ?Answer ?Proof',
            'rotate_left',
        ] + ([
            'intros ' + ' '.join([v.name or '_' for v in sample.hypotheses])
                ] if len(Phi) > 0 else [])

        self.formal_problem_framework = 'example :\n' + \
            (('∀ ' + '\n  '.join(V) + ',\n') if len(V) > 0 else '\n') + \
            f'∃ (answer : {self.sample.formal_answer_type}),' + \
            (('\n∀ ' + '\n  '.join(Phi) + ',\n') if len(Phi) > 0 else '\n') + \
            ' ∧ '.join(Psi) + '\n:= by\n' + \
            '\n'.join(init_steps) + \
            '\nall_goals sorry'
    
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        assert len(init_solution_state.goals) == 2 and [g.name for g in init_solution_state.goals] == ['Proof', 'Answer'], f'Invalid solving_state:\n{init_solution_state}'
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'Answer']
        assert len(answer_mvarId) == 1, f'Invalid answer_mvarId in init_solution_state: {[init_solution_state]}'
        self.answer_mvarId = answer_mvarId[0]

        return init_solution_state, init_steps
            
    init_solving_state = to_sync(init_solving_state_async)

    async def prove_eq_async(self, submission: str) -> Optional[str]:
        await self.server.restart_async()
        rpe_proof = '''  try ext
  try funext
  repeat''' + ' {\n' + '\n'.join(('    try ' + tac + ';') for tac in RPE_TACTICS) + '\n  }'
        rpe_code = 'example' + \
            ((' ' + format_variable_sequence(self.sample.independent_variables) + '\n') if len(self.sample.independent_variables) > 0 else '\n') + \
            f' (answer : {self.sample.formal_answer_type}) : (answer = (\n{submission} : {self.sample.formal_answer_type}\n)) ↔ (\n{self.sample.formal_answer}\n)' + ' := by\n' + rpe_proof
        
        try:
            units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+rpe_code)
            assert 'error' not in str([x.messages for x in units]) and all((u.goal_state is None or u.goal_state.is_solved) for u in units), f'formal_problem_framework:\n{(self.sample.header or OPEN_HEADER)}\n{rpe_code}\nmessage:\n{str([x.messages for x in units])}'
            return rpe_proof
        except:
            return None

    prove_eq = to_sync(prove_eq_async)


# Answer as Prop
class PropSolvingServer(BaseSolvingServer):
    async def init_solving_state_async(self, sample: Optional[FormalProblem]=None) -> Tuple[GoalState, List[str]]:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        V = [f"  ({v.name or '_'} : {v.t})" for v in sample.independent_variables]
        Phi = [f"  ({v.name or '_'} : {v.t})" for v in sample.hypotheses]
        Psi = sample.conclusions

        # From FPS to D-FPS
        V.append(f'   (answer : {self.sample.formal_answer_type})')
        Psi = '(' + ' ∧ '.join(Psi) + f' ↔ {H_SUBMISSION_NAME})'

        init_steps = [
            'intros ' + ' '.join([v.name or '_' for v in self.sample.independent_variables] + ['answer']),
            'apply Exists.intro ?Answer ?Proof',
            'rotate_left'
        ] + \
            (['intros ' + ' '.join([v.name or '_' for v in sample.hypotheses])] if len(Phi) > 0 else []) + \
        [
            'refine @Iff.intro ?_ _ ?Forward ?Backward',
            'rotate_left',
            f'intros {H_SUBMISSION_NAME}',
            'rotate_right',
            f'intros {H_CONCLUSION_NAME}'
        ]

        self.formal_problem_framework = 'example :\n' + \
            (('∀ ' + '\n  '.join(V) + ',\n') if len(V) > 0 else '\n') + \
            f'∃ ({H_SUBMISSION_NAME} : Prop),' + \
            (('\n∀ ' + '\n  '.join(Phi) + ',\n') if len(Phi) > 0 else '\n') + \
            Psi + '\n:= by\n' + \
            '\n'.join(init_steps) + \
            '\nall_goals sorry'
    
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        assert len(init_solution_state.goals) == 3 and [g.name for g in init_solution_state.goals] == ['Forward', 'Backward', 'Answer'], f'Invalid solving_state:\n{init_solution_state}'
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'Answer']
        assert len(answer_mvarId) == 1, f'Invalid answer_mvarId in init_solution_state: {[init_solution_state]}'
        self.answer_mvarId = answer_mvarId[0]

        return init_solution_state, init_steps
        
    init_solving_state = to_sync(init_solving_state_async)

    async def init_forward_solving_state_async(self, sample: Optional[FormalProblem]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        init_steps = [
            'intros ' + ' '.join([v.name or '_' for v in self.sample.independent_variables] + ['answer']),
            'apply Exists.intro ?Answer ?Proof',
            'rotate_left',
        ] + ([
            'intros ' + ' '.join([v.name or '_' for v in sample.hypotheses])
        ] if len(sample.hypotheses) > 0 else [])
        
        self.formal_problem_framework = 'example :\n' + \
            (('∀ ' + format_variable_sequence(self.sample.independent_variables) + ',\n') if len(self.sample.independent_variables) > 0 else '') + \
            f'∀ (answer : {self.sample.formal_answer_type}), ∃ ({H_SUBMISSION_NAME} : Prop), \n' + \
            (('∀ ' + format_variable_sequence(self.sample.hypotheses) + ',\n') if len(self.sample.hypotheses) > 0 else '') + \
            f'({H_SUBMISSION_NAME})' + '\n:= by\n' + \
            '\n'.join(init_steps) + \
            '\nall_goals sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        assert [g.name for g in init_solution_state.goals] == ['Proof', 'Answer'], f'Invalid solving_state:\n{init_solution_state}'
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'Answer']
        assert len(answer_mvarId) == 1, f'Invalid answer_mvarId in init_solution_state: {[init_solution_state]}'
        self.answer_mvarId = answer_mvarId[0]
        
        return init_solution_state, init_steps
        
    init_forward_solving_state = to_sync(init_forward_solving_state_async)

    async def prove_eq_async(self, submission: str) -> Optional[str]:
        await self.server.restart_async()
        rpe_proof = '''  try ext
  try funext
  repeat''' + ' {\n' + '\n'.join(('    try ' + tac + ';') for tac in RPE_TACTICS) + '\n  }'
        rpe_code = 'example' + \
            ((' ' + format_variable_sequence(self.sample.independent_variables) + '\n') if len(self.sample.independent_variables) > 0 else '\n') + \
            f' (answer : {self.sample.formal_answer_type}) : (\n{submission}\n) ↔ (\n{self.sample.formal_answer}\n)' + ' := by\n' + rpe_proof
        
        try:
            units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+rpe_code)
            assert 'error' not in str([x.messages for x in units]) and all((u.goal_state is None or u.goal_state.is_solved) for u in units), f'formal_problem_framework:\n{(self.sample.header or OPEN_HEADER)}\n{rpe_code}\nmessage:\n{str([x.messages for x in units])}'
            return rpe_proof
        except:
            return None

    prove_eq = to_sync(prove_eq_async)
