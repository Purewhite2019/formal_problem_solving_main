from dataclasses import dataclass, field, fields
from typing import Tuple, List, Dict, Optional
from enum import Enum
import json
import regex as re

from loguru import logger
from dacite import from_dict
import msgspec

from common.utils import parse_expr, normalize_draft, replace_span, remove_comments
from common.constants import Expr


@dataclass(frozen=True)
class Variable:
    t: Expr
    v: Optional[Expr] = None
    name: Optional[str] = None

    @staticmethod
    def parse(payload: Dict):
        name = payload.get("userName")
        t = parse_expr(payload["type"])
        v = payload.get("value")
        if v:
            v = parse_expr(v)
        return Variable(t, v, name)

    def __str__(self):
        """
        :meta public:
        """
        result = self.name if self.name else "_"
        result += f" : {self.t}"
        if self.v:
            result += f" := {self.v}"
        return result

    def serialize(self) -> Dict:
        return {
            't': self.t,
            'v': self.v,
            'name': self.name,
        }

@dataclass(frozen=True)
class Goal:
    variables: List[Variable]
    target: Expr
    sibling_dep: List[int] = field(default_factory=lambda: [])
    name: Optional[str] = None
    is_conversion: bool = False

    @staticmethod
    def sentence(target: Expr):
        """
        :meta public:
        """
        return Goal(variables=[], target=target)

    @staticmethod
    def parse(payload: Dict, sibling_map: Dict[str, int]):
        name = payload.get("userName")
        variables = [Variable.parse(v) for v in payload["vars"]]
        target = parse_expr(payload["target"])
        is_conversion = payload["isConversion"]

        dependents = payload["target"]["dependentMVars"]
        sibling_dep = [sibling_map[d] for d in dependents if d in sibling_map]

        return Goal(variables, target, sibling_dep, name, is_conversion)

    def __str__(self):
        """
        :meta public:
        """
        output = '' if self.name is None else 'case ' + self.name + '\n'
        if len(self.variables) > 0:
            vars_to_format = [v for v in self.variables]
            while len(vars_to_format) > 0:
                for i in range(len(vars_to_format)):
                    if i + 1 == len(vars_to_format) or not (vars_to_format[i].t == vars_to_format[i+1].t and vars_to_format[i].v is None and vars_to_format[i+1].v is None):
                        break
                if i == 0:
                    output += str(vars_to_format[0]) + '\n'
                    vars_to_format.pop(0)
                else:
                    output += ' '.join([v.name if v.name is not None else "_" for v in vars_to_format[:i+1]]) + f' : {vars_to_format[0].t}\n'
                    vars_to_format = vars_to_format[i+1:]
        output += ("|" if self.is_conversion else "⊢") + ' ' + self.target
        return output

    def serialize(self) -> Dict:
        return {
            'variables': [v.serialize() for v in self.variables],
            'target': self.target,
            'sibling_dep': self.sibling_dep,
            'name': self.name,
            'is_conversion': self.is_conversion
        }

@dataclass(frozen=True)
class GoalState:
    state_id: int
    goals: List[Goal]
    payload: Dict
    _sentinel: List[int]

    def serialize(self) -> Dict:
        return {
            'state_id': self.state_id,
            'goals': [g.serialize() for g in self.goals],
            'payload': self.payload,
            '_sentinel': [],
        }

    def __del__(self):
        self._sentinel.append(self.state_id)

    @property
    def is_solved(self) -> bool:
        """
        WARNING: Does not handle dormant goals.

        :meta public:
        """
        return not self.goals

    @staticmethod
    def parse_inner(state_id: int, goals: List, payload: Dict, _sentinel: List[int]):
        goal_names = { g["name"]: i for i, g in enumerate(goals) }
        goals = [Goal.parse(g, goal_names) for g in goals]
        return GoalState(state_id, goals, payload, _sentinel)
    @staticmethod
    def parse(payload: Dict, _sentinel: List[int]):
        return GoalState.parse_inner(payload["nextStateId"], payload["goals"], payload, _sentinel)

    def __str__(self):
        """
        :meta public:
        """
        return '\n'.join([
            str(g) for g in self.goals
        ])
    
    @property
    def first_goal_str(self) -> Optional[str]:
        if len(self.goals) == 0:
            return None
        
        for cur_i, cur_g in enumerate(self.goals):
            if len(cur_g.sibling_dep) == 0:
                break
        assert cur_i < len(self.goals), "All goals have nonempty sibling_dep in state " + '\n'.join([repr(g) for g in self.goals])

        sibling_deps = [i for (i, g) in enumerate(self.goals) if cur_i in g.sibling_dep]
        return '\n'.join([
            str(self.goals[g]) for g in [cur_i] + sibling_deps
        ])

    @property
    def first_goal_id(self) -> Optional[str]:
        if len(self.goals) == 0:
            return None
        
        for cur_i, cur_g in enumerate(self.goals):
            if len(cur_g.sibling_dep) == 0:
                break
        assert cur_i < len(self.goals), "All goals have nonempty sibling_dep in state " + '\n'.join([repr(g) for g in self.goals])

        return cur_i


@dataclass(frozen=True)
class TacticHave:
    """
    The `have` tactic, equivalent to
    ```lean
    have {binder_name} : {branch} := ...
    ```
    """
    branch: str
    binder_name: Optional[str] = None
    def __str__(self) -> str:
        return f"have {self.binder_name} : {self.branch} := sorry" if self.binder_name is not None else f"have : {self.branch} := sorry"


@dataclass(frozen=True)
class TacticLet:
    """
    The `let` tactic, equivalent to
    ```lean
    let {binder_name} : {branch} := ...
    ```
    """
    branch: str
    binder_name: Optional[str] = None
    def __str__(self) -> str:
        return f"let {self.binder_name} : {self.branch} := sorry" if self.binder_name is not None else f"let : {self.branch} := sorry"

@dataclass(frozen=True)
class TacticCalc:
    """
    The `calc` tactic, equivalent to
    ```lean
    calc {step} := ...
    ```
    You can use `_` in the step.
    """
    step: str

@dataclass(frozen=True)
class TacticExpr:
    """
    Assigns an expression to the current goal
    """
    expr: str

@dataclass(frozen=True)
class TacticDraft:
    """
    Assigns an expression to the current goal
    """
    expr: str

Tactic = str

@dataclass(frozen=True)
class TacticInvocation:
    """
    One tactic invocation with the before/after goals extracted from Lean source
    code.
    """
    before: List[str]
    after: List[str]
    tactic: str

    @staticmethod
    def parse(payload: Dict):
        return TacticInvocation(
            before=payload["goalBefore"],
            after=payload["goalAfter"],
            tactic=payload["tactic"],
        )

    def serialize(self) -> Dict:
        return {
            'before': self.before,
            'after': self.after,
            'tactic': self.tactic,
        }

@dataclass(frozen=True)
class CompilationUnit:

    # Byte boundaries [begin, end[ of each compilation unit.
    i_begin: int
    i_end: int

    messages: List[str] = field(default_factory=list)

    invocations: Optional[list[TacticInvocation]] = None
    # If `goal_state` is none, maybe error has occurred. See `messages`
    goal_state: Optional[GoalState] = None
    goal_src_boundaries: Optional[list[Tuple[int, int]]] = None

    new_constants: Optional[list[str]] = None

    @staticmethod
    def parse(payload: Dict, goal_state_sentinel=None):
        i_begin = payload["boundary"][0]
        i_end = payload["boundary"][1]
        messages = payload["messages"]

        if (invocation_payload := payload.get("invocations")) is not None:
            invocations = [
                TacticInvocation.parse(i) for i in invocation_payload
            ]
        else:
            invocations = None

        if (state_id := payload.get("goalStateId")) is not None:
            goal_state = GoalState.parse_inner(int(state_id), payload["goals"], payload, goal_state_sentinel)
            goal_src_boundaries = payload["goalSrcBoundaries"]
        else:
            goal_state = None
            goal_src_boundaries = None

        new_constants = payload.get("newConstants")

        return CompilationUnit(
            i_begin,
            i_end,
            messages,
            invocations,
            goal_state,
            goal_src_boundaries,
            new_constants
        )
    
    def serialize(self) -> Dict:
        return {
            'i_begin': self.i_begin,
            'i_end': self.i_end,
            'messages': self.messages,
            'invocations': None if self.invocations is None else [ivc.serialize() for ivc in self.invocations],
            'goal_state': None if self.goal_state is None else self.goal_state.serialize(),
            'goal_src_boundaries': None if self.goal_src_boundaries is None else self.goal_src_boundaries,
            'new_constants': None if self.new_constants is None else self.new_constants,
        }

@dataclass(frozen=True)
class ProofSearchState:
    state: GoalState
    parent: Optional['ProofSearchState']
    last_step: Optional[Tuple[int, Tactic]]
    cost: float

    def __lt__(self, other: 'ProofSearchState') -> bool:
        assert isinstance(other, ProofSearchState)
        return self.cost < other.cost
    
    @property
    def tactic_history(self) -> List[Tactic]:
        return ([] if self.parent is None else self.parent.tactic_history) + ([] if self.last_step is None else [self.last_step])

    @property
    def state_history(self) -> List[GoalState]:
        return ([] if self.parent is None else self.parent.state_history) + [self.state]

@dataclass
class FormalProblem:
    # Informal information
    informal_problem: Optional[str]=None
    informal_answer: Optional[str]=None
    informal_solution: Optional[str]=None
    
    # Formal problem
    header: Optional[str]=None
    independent_variables: List[Variable]=field(default_factory=list)
    formal_answer: Optional[str]=None
    formal_answer_type: Optional[str]=None
    hypotheses: List[Variable]=field(default_factory=list)
    conclusions: List[str]=field(default_factory=list)
    
    # Meta information
    metainfo: Dict=field(default_factory=dict)

    def __post_init__(self):
        for (i, v) in enumerate(self.independent_variables):
            if isinstance(v, dict):
                self.independent_variables[i] = from_dict(Variable, v)
        for (i, v) in enumerate(self.hypotheses):
            if isinstance(v, dict):
                self.hypotheses[i] = from_dict(Variable, v)

    @classmethod
    def from_kwargs(cls, **kwargs):
        cls_fields = [f.name for f in fields(cls)]

        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                if name == 'metainfo':
                    assert isinstance(val, Dict), 'field `metainfo` should be a `Dict`'
                    new_args |= val
                else:
                    native_args[name] = val
            else:
                new_args[name] = val

        ret = cls(metainfo=new_args, **native_args)

        return ret

    def serialize(self) -> Dict:
        return {
            'informal_problem': self.informal_problem,
            'informal_answer': self.informal_answer,
            'header': self.header,
            'independent_variables': [v.serialize() for v in self.independent_variables],
            'formal_answer': self.formal_answer,
            'formal_answer_type': self.formal_answer_type,
            'hypotheses': [v.serialize() for v in self.hypotheses],
            'conclusions': self.conclusions,
            'metainfo': self.metainfo,
        }
