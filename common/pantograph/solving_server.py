
import collections as C
import functools as F
import itertools as I
from typing import Optional, Union, List, Dict, Tuple, Set, Iterable

from common.constants import CORE_OPTIONS, OPEN_HEADER, RPE_TACTICS, H_SUBMISSION_NAME
from common.utils import format_variable_sequence, inplace_add, to_sync
from common.pantograph.server import Server, TacticFailure
from common.pantograph.dataclasses import GoalState, FormalProblem, FormalProblem


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

    async def load_problem_async(self, sample: Optional[FormalProblem]) -> None:
        if sample is None:
            return None
        assert isinstance(sample, FormalProblem)
        self.sample = sample
        self.server = await Server.create(
            imports=self.imports,
            project_path=self.project_path,
            core_options=CORE_OPTIONS,
            timeout=300,
        )
        return None

    async def init_backward_state_async(self, sample: Optional[FormalProblem]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.formal_problem_framework = 'example : ∀ ' + \
            ((format_variable_sequence(self.sample.intros) + ' ') if len(self.sample.intros) > 0 else '') + \
            f'(answer : {self.sample.formal_answer_type}) ({H_SUBMISSION_NAME}: {self.sample.formal_answer}) ' + \
            ((format_variable_sequence(self.sample.outros[:-1]) + ' ') if len(self.sample.outros) > 1 else '') + \
            f', ({self.sample.outros[-1].t})\n:= sorry\n'

        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        return units[-1].goal_state
            
    init_backward_state = to_sync(init_backward_state_async)

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
        tac_history = []
        
        self.answer_mvarId = None
        self.formal_problem_framework = 'example :' + \
            ((' ∀ ' + format_variable_sequence(self.sample.intros) + ',\n') if len(self.sample.intros) > 0 else '\n') + \
            f'∃ (answer : {self.sample.formal_answer_type}),' + \
            (('\n∀ ' + format_variable_sequence(self.sample.outros[:-1]) + ',\n') if len(self.sample.outros) > 1 else '\n') + \
            f'({self.sample.outros[-1].t})\n:= sorry\n'

        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:\n{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        
        if len(self.sample.intros) > 0:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, inplace_add('intros ' + ' '.join([v.name or '_' for v in self.sample.intros]), tac_history))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, inplace_add('apply Exists.intro', tac_history))
        
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'w']
        assert len(answer_mvarId) == 1
        self.answer_mvarId = answer_mvarId[0]
        
        if len(self.sample.outros) > 1:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, inplace_add('intros ' + ' '.join([v.name or '_' for v in self.sample.outros[:-1]]), tac_history))
        
        assert len(init_solution_state.goals) == 2 and [g.name for g in init_solution_state.goals] == ['h', 'w'], f'Invalid solving_state:\n{init_solution_state}'

        return init_solution_state, tac_history
            
    init_solving_state = to_sync(init_solving_state_async)

    async def check_rpe_async(self, submission: str) -> Optional[str]:
        await self.server.restart_async()
        rpe_code = 'example' + \
            ((' ' + format_variable_sequence(self.sample.intros) + '\n') if len(self.sample.intros) > 0 else '\n') + \
            f' (answer : {self.sample.formal_answer_type}) : (answer = (\n{submission} : {self.sample.formal_answer_type}\n)) ↔ (\n{self.sample.formal_answer}\n)' + ' := by\n' + '\n'.join(['try ' + tac for tac in RPE_TACTICS])

        try:
            units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER) + '\n' + rpe_code)
            assert len(units) >= 1, f'len(units)={len(units)}'
            assert 'error' not in str([x.messages for x in units]), str([x.messages for x in units])
            assert units[-1].goal_state is None or units[-1].goal_state.is_solved, 'Unsolved units[-1].goal_state:\n' + str(units[-1].goal_state)
            return RPE_TACTICS
        except:
            return None

    check_rpe = to_sync(check_rpe_async)


# Answer as Prop
class PropSolvingServer(BaseSolvingServer):
    async def init_solving_state_async(self, sample: Optional[FormalProblem]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        self.formal_problem_framework = 'example :' + \
            ((' ∀ ' + format_variable_sequence(self.sample.intros) + ',\n') if len(self.sample.intros) > 0 else '\n') + \
            f'∀ (answer : {self.sample.formal_answer_type}), ∃ ({H_SUBMISSION_NAME} : Prop), ' + \
            (('\n∀ ' + format_variable_sequence(self.sample.outros[:-1]) + ',\n') if len(self.sample.outros) > 1 else '\n') + \
            f'(({self.sample.outros[-1].t}) ↔ {H_SUBMISSION_NAME})\n:= sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        
        if len(self.sample.intros) > 0:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.intros]))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros answer')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'apply Exists.intro')
        
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'w']
        assert len(answer_mvarId) == 1
        self.answer_mvarId = answer_mvarId[0]
        
        if len(self.sample.outros) > 1:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.outros[:-1]]))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'constructor')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 1, f'intros {H_SUBMISSION_NAME}')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 1, 'intros ' + self.sample.outros[-1].name)
        
        assert len(init_solution_state.goals) == 3 and [g.name for g in init_solution_state.goals] == ['h.mp', 'h.mpr', 'w'], f'Invalid solving_state:\n{init_solution_state}'
    
        return init_solution_state
        
    init_solving_state = to_sync(init_solving_state_async)

    async def init_forward_reasoning_state_async(self, sample: Optional[FormalProblem]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.formal_problem_framework = 'example ' + \
            ((format_variable_sequence(self.sample.intros) + '\n') if len(self.sample.intros) > 0 else '\n') + \
            f'(answer : {self.sample.formal_answer_type}) ' + \
            (('\n' + format_variable_sequence(self.sample.outros[:]) + '\n') if len(self.sample.outros) > 0 else '\n') + \
            f': False := sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        assert len(init_solution_state.goals) == 1 and [g.name for g in init_solution_state.goals] == [None], f'Invalid solving_state:\n{init_solution_state}'
        
        return init_solution_state
        
    init_forward_reasoning_state = to_sync(init_forward_reasoning_state_async)

    async def check_rpe_async(self, submission: str) -> Optional[str]:
        await self.server.restart_async()
        rpe_code = 'example' + \
            ((' ' + format_variable_sequence(self.sample.intros) + '\n') if len(self.sample.intros) > 0 else '\n') + \
            f' (answer : {self.sample.formal_answer_type}) : (\n{submission}\n) ↔ (\n{self.sample.formal_answer}\n)' + ' := by\n' + '\n'.join(['try ' + tac for tac in RPE_TACTICS])

        try:
            units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER) + '\n' + rpe_code)
            assert len(units) >= 1, f'len(units)={len(units)}'
            assert 'error' not in str([x.messages for x in units]), str([x.messages for x in units])
            assert units[-1].goal_state is None or units[-1].goal_state.is_solved, 'Unsolved units[-1].goal_state:\n' + str(units[-1].goal_state)
            return RPE_TACTICS
        except:
            return None

    check_rpe = to_sync(check_rpe_async)
