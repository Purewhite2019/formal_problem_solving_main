import json, pexpect, unittest, os
from typing import Union, List, Dict, Optional, Any, Tuple
from pathlib import Path
import asyncio
import warnings
import signal
import functools as F

import nest_asyncio
nest_asyncio.apply()
from loguru import logger
import numpy as np
from async_lru import alru_cache

from common.utils import to_sync, Spawn
from common.constants import DEFAULT_CORE_OPTIONS, REPL_TIMEOUT, REPL_MAXREAD
from common.pantograph.dataclasses import (
    parse_expr,
    Expr,
    Variable,
    Goal,
    GoalState,
    Tactic,
    TacticHave,
    TacticLet,
    TacticCalc,
    TacticExpr,
    TacticDraft,
    CompilationUnit,
    TacticInvocation,
)

def _get_proc_cwd():
    return Path(__file__).parent
def _get_proc_path():
    return _get_proc_cwd() / "pantograph-repl"

@alru_cache(maxsize=128)
async def check_output(*args, **kwargs):
    p = await asyncio.create_subprocess_exec(
        *args, 
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        stdout_data, stderr_data = await p.communicate()
    if p.returncode == 0:
        return stdout_data

@alru_cache(maxsize=128)
async def get_lean_path_async(project_path):
    """
    Extracts the `LEAN_PATH` variable from a project path.
    """
    p = await check_output(
        'lake', 'env', 'printenv', 'LEAN_PATH',
        cwd=project_path,
    )
    return p

get_lean_path = to_sync(get_lean_path_async)

class TacticFailure(Exception):
    """
    Indicates a tactic failed to execute
    """
class ServerError(Exception):
    """
    Indicates a logical error in the server.
    """

class Server:
    """
    Main interaction instance with Pantograph
    """
    
    def __init__(self,
                 imports: List[str]=["Init"],
                 project_path: Optional[str]=None,
                 lean_path: Optional[str]=None,
                 # Options for executing the REPL.
                 # Set `{ "automaticMode" : False }` to handle resumption by yourself.
                 options: Dict[str, Any]={},
                 core_options: List[str]=DEFAULT_CORE_OPTIONS,
                 timeout: int=REPL_TIMEOUT,
                 maxread: int=REPL_MAXREAD,
                 _sync_init: bool=True):
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.project_path = project_path if project_path else _get_proc_cwd()
        if _sync_init and project_path and not lean_path:
            lean_path = get_lean_path(project_path)
        self.lean_path = lean_path
        self.maxread = maxread
        self.proc_path = _get_proc_path()

        self.options = options
        self.core_options = core_options
        self.args = " ".join(imports + [f'--{opt}' for opt in core_options])
        self.proc = None
        if _sync_init:
            self.restart()

        # List of goal states that should be garbage collected
        self.to_remove_goal_states = []

    @classmethod
    async def create(cls,
                 imports: List[str]=["Init"],
                 project_path: Optional[str]=None,
                 lean_path: Optional[str]=None,
                 # Options for executing the REPL.
                 # Set `{ "automaticMode" : False }` to handle resumption by yourself.
                 options: Dict[str, Any]={},
                 core_options: List[str]=DEFAULT_CORE_OPTIONS,
                 timeout: int=REPL_TIMEOUT,
                 maxread: int=REPL_MAXREAD,
                 start:bool=True) -> 'Server':
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self = cls(
            imports,
            project_path,
            lean_path,
            options,
            core_options,
            timeout,
            maxread,
            _sync_init=False
        )
        if project_path and not lean_path:
            lean_path = await get_lean_path_async(project_path)
        self.lean_path = lean_path
        if start:
            await self.restart_async()
        return self

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._close()

    def __del__(self):
        self._close()
        # if hasattr(self, 'timing_history'):
        #     logger.debug('Timing history: ' + '\n'.join([f'{k} : {np.mean(v):.3f}±{np.std(v):.3f}' for k, v in self.timing_history.items()]))
    
    def _close(self):
        if self.proc is not None:
            try:
                if self.proc.async_pw_transport:
                    self.proc.async_pw_transport[1].close()
                self.proc.close()
                if self.proc.isalive():
                    self.proc.terminate(force=True)
            except:
                pass
            self.proc = None

    def is_automatic(self):
        """
        Check if the server is running in automatic mode
        """
        return self.options.get("automaticMode", True)

    # @timeit
    async def restart_async(self):
        if self.proc is not None:
            self._close()
        env = os.environ
        if self.lean_path:
            env = env | {'LEAN_PATH': self.lean_path}

        def timeout_handler(signum, frame):
            raise TimeoutError("Server Spwan() timed out")

        if self.timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

        self.proc = Spawn(
            f"{self.proc_path} {self.args}",
            encoding="utf-8",
            maxread=self.maxread,
            timeout=self.timeout,
            cwd=self.project_path,
            env=env,
        )
        self.proc.setecho(False) # Do not send any command before this.
        if self.timeout > 0:
            signal.alarm(0)

        try:
            ready = await self.proc.readline_async() # Reads the "ready."
            assert ready.rstrip() == "ready.", f"Server failed to emit ready signal: {ready}; Maybe the project needs to be rebuilt"
        except pexpect.exceptions.TIMEOUT as exc:
            raise ServerError("Server failed to emit ready signal in time") from exc

        if self.options:
            await self.run_async("options.set", self.options)

        await self.run_async('options.set', {'printDependentMVars': True})

    restart = to_sync(restart_async)

    async def run_async(self, cmd, payload):
        """
        Runs a raw JSON command. Preferably use one of the commands below.

        :meta private:
        """
        assert self.proc
        s = json.dumps(payload, ensure_ascii=False)
        
        self.proc.sendline(f"{cmd} {s}")
        
        try:
            line = await self.proc.readline_async()
            try:
                obj = json.loads(line)
                if obj.get("error") == "io":
                    # The server is dead
                    self._close()
                return obj
            except Exception as e:
                # self.proc.sendeof()
                remainder = await self.proc.read_async()
                # self._close()
                raise ServerError(f"Cannot decode: {line}\n{remainder}") from e
        except pexpect.exceptions.TIMEOUT as exc:
            self._close()
            return {"error": "timeout", "message": str(exc)}

    run = to_sync(run_async)

    # @timeit
    async def gc_async(self):
        """
        Garbage collect deleted goal states to free up memory.
        """
        if not self.to_remove_goal_states:
            return
        result = await self.run_async('goal.delete', {'stateIds': self.to_remove_goal_states})
        self.to_remove_goal_states.clear()
        if "error" in result:
            raise ServerError(result)

    gc = to_sync(gc_async)

    # @timeit
    async def expr_type_async(self, expr: Expr) -> Expr:
        """
        Evaluate the type of a given expression. This gives an error if the
        input `expr` is ill-formed.
        """
        result = await self.run_async('expr.echo', {"expr": expr})
        if "error" in result:
            raise ServerError(result)
        return parse_expr(result["type"])

    expr_type = to_sync(expr_type_async)

    # @timeit
    async def goal_start_async(self, expr: Expr) -> GoalState:
        """
        Create a goal state with one root goal, whose target is `expr`
        """
        result = await self.run_async('goal.start', {"expr": str(expr)})
        if "error" in result:
            print(f"Cannot start goal: {expr}")
            raise ServerError(result)
        return GoalState(
            state_id=result["stateId"],
            goals=[Goal.sentence(expr)],
            payload=result,
            _sentinel=self.to_remove_goal_states,
        )

    goal_start = to_sync(goal_start_async)

    # @timeit
    async def goal_tactic_async(self, state: GoalState, goal_id: int, tactic: Tactic) -> GoalState:
        """
        Execute a tactic on `goal_id` of `state`
        """
        args = {"stateId": state.state_id, "goalId": goal_id}
        if isinstance(tactic, str):
            args["tactic"] = tactic
        elif isinstance(tactic, TacticHave):
            args["have"] = tactic.branch
            if tactic.binder_name:
                args["binderName"] = tactic.binder_name
        elif isinstance(tactic, TacticLet):
            args["let"] = tactic.branch
            if tactic.binder_name:
                args["binderName"] = tactic.binder_name
        elif isinstance(tactic, TacticCalc):
            args["calc"] = tactic.step
        elif isinstance(tactic, TacticExpr):
            args["expr"] = tactic.expr
        elif isinstance(tactic, TacticDraft):
            args["draft"] = tactic.expr
        else:
            raise TacticFailure(f"Invalid tactic type: {tactic}")
        result = await self.run_async('goal.tactic', args)
        if "error" in result:
            raise ServerError(result)
        if "tacticErrors" in result:
            raise TacticFailure(result)
        if "parseError" in result:
            raise TacticFailure(result)
        return GoalState.parse(result, self.to_remove_goal_states)

    goal_tactic = to_sync(goal_tactic_async)

    # @timeit
    async def goal_conv_begin_async(self, state: GoalState, goal_id: int) -> GoalState:
        """
        Start conversion tactic mode on one goal
        """
        result = await self.run_async('goal.tactic', {"stateId": state.state_id, "goalId": goal_id, "conv": True})
        if "error" in result:
            raise ServerError(result)
        if "tacticErrors" in result:
            raise ServerError(result)
        if "parseError" in result:
            raise ServerError(result)
        return GoalState.parse(result, self.to_remove_goal_states)

    goal_conv_begin = to_sync(goal_conv_begin_async)

    # @timeit
    async def goal_conv_end_async(self, state: GoalState) -> GoalState:
        """
        Exit conversion tactic mode on all goals
        """
        result = await self.run_async('goal.tactic', {"stateId": state.state_id, "goalId": 0, "conv": False})
        if "error" in result:
            raise ServerError(result)
        if "tacticErrors" in result:
            raise ServerError(result)
        if "parseError" in result:
            raise ServerError(result)
        return GoalState.parse(result, self.to_remove_goal_states)

    goal_conv_end = to_sync(goal_conv_end_async)

    # @timeit
    async def tactic_invocations_async(self, file_name: Union[str, Path]) -> List[CompilationUnit]:
        """
        Collect tactic invocation points in file, and return them.
        """
        result = await self.run_async('frontend.process', {
            'fileName': str(file_name),
            'invocations': True,
            "sorrys": False,
            "newConstants": False,
            "typeErrorsAsGoals": False,
        })
        if "error" in result:
            raise ServerError(result)

        units = [CompilationUnit.parse(payload) for payload in result['units']]
        return units

    tactic_invocations = to_sync(tactic_invocations_async)

    # @timeit
    async def parse_proof_async(self, content: str) -> List[CompilationUnit]:
        """
        Collect tactic invocation points in file, and return them.
        """
        result = await self.run_async('frontend.process', {
            'file': content,
            'invocations': True,
            "sorrys": False,
            "newConstants": False,
            "typeErrorsAsGoals": False,
        })
        if "error" in result:
            raise ServerError(result)

        units = [CompilationUnit.parse(payload) for payload in result['units']]
        return units

    parse_proof = to_sync(parse_proof_async)

    # @timeit
    async def load_sorry_async(self, content: str) -> List[CompilationUnit]:
        """
        Executes the compiler on a Lean file. For each compilation unit, either
        return the gathered `sorry` s, or a list of messages indicating error.
        """
        result = await self.run_async('frontend.process', {
            'file': content,
            'invocations': False,
            "sorrys": True,
            "newConstants": False,
            "typeErrorsAsGoals": False,
        })
        if "error" in result:
            raise ServerError(result)

        units = [
            CompilationUnit.parse(payload, goal_state_sentinel=self.to_remove_goal_states)
            for payload in result['units']
        ]
        return units

    load_sorry = to_sync(load_sorry_async)

    # @timeit
    async def env_add_async(self, name: str, t: Expr, v: Expr, is_theorem: bool = True):
        result = await self.run_async('env.add', {
            "name": name,
            "type": t,
            "value": v,
            "isTheorem": is_theorem,
            "typeErrorsAsGoals": False,
        })
        if "error" in result:
            raise ServerError(result)
    
    env_add = to_sync(env_add_async)
    
    # @timeit
    async def env_inspect_async(
            self,
            name: str,
            print_value: bool = False,
            print_dependency: bool = False) -> Dict:
        result = await self.run_async('env.inspect', {
            "name": name,
            "value": print_value,
            "dependency": print_dependency,
            "source": True,
        })
        if "error" in result:
            raise ServerError(result)
        return result
    env_inspect = to_sync(env_inspect_async)

    # @timeit
    async def env_save_async(self, path: str):
        result = await self.run_async('env.save', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result)
    env_save = to_sync(env_save_async)
    
    # @timeit
    async def env_load_async(self, path: str):
        result = await self.run_async('env.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result)
    
    env_load = to_sync(env_load_async)

    # @timeit
    async def goal_save_async(self, goal_state: GoalState, path: str):
        result = await self.run_async('goal.save', {
            "id": goal_state.state_id,
            "path": path,
        })
        if "error" in result:
            raise ServerError(result)
    
    goal_save = to_sync(goal_save_async)
    
    # @timeit
    async def goal_load_async(self, path: str) -> GoalState:
        result = await self.run_async('goal.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result)
        state_id = result['id']
        result = await self.run_async('goal.print', {
            'stateId': state_id,
            'goals': True,
        })
        if "error" in result:
            raise ServerError(result)
        return GoalState.parse_inner(state_id, result['goals'], result, self.to_remove_goal_states)
    
    goal_load = to_sync(goal_load_async)

    # @timeit
    async def goal_print_async(self, goal_state: GoalState, rootExpr: bool, parentExpr: bool, goals: bool, extraMVars: List[str]=[]):
        result = await self.run_async('goal.print', {
            "stateId": goal_state.state_id,
            "rootExpr": rootExpr,
            "parentExpr" : parentExpr,
            "goals" : goals,
            "extraMVars": extraMVars
            
        })
        if "error" in result:
            raise ServerError(result)
        return result
    
    goal_print = to_sync(goal_print_async)

    # @timeit
    async def reset_async(self):
        result = await self.run_async('reset', dict())
        if "error" in result:
            raise ServerError(result)
        return result
    
    reset = to_sync(reset_async)

def get_version():
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"],
                          stdout=subprocess.PIPE,
                          cwd=_get_proc_cwd()) as p:
        return p.communicate()[0].decode('utf-8').strip()


def record_server_error(func):
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            return result
        except (ServerError, TimeoutError, pexpect.exceptions.TIMEOUT) as e:
            self.count = float('inf')
            logger.debug(f'PersistentServer({self.tag}): Captured {type(e)}, restart at the next action.')
            raise e
    return wrapper

class AutoRecoverServer:
    def __init__(self, max_count: int=1024, tag: str='', _sync_init: bool=True, *args, **kwargs):
        self.max_count = max_count
        self.tag = tag
        self.server_args = args
        self.server_kwargs = kwargs
        self.count = float('inf')
        if _sync_init:
            self.check_restart()

    async def check_restart_async(self) -> None:
        if self.count < self.max_count: # Assuming if server is dead, self.count is set to float('inf'). 
            return
        
        logger.debug(f'PersistentServer({self.tag}): Restarting...')
        self.server = await Server.create(*self.server_args, **self.server_kwargs)

        self.count = 0
    
    check_restart = to_sync(check_restart_async)

    @record_server_error
    async def load_sorry_async(self, code: str) -> GoalState :
        await self.check_restart_async()
        self.count += 1

        return await self.server.load_sorry_async(code)

    def is_automatic(self):
        """
        Check if the server is running in automatic mode
        """
        return self.server.is_automatic()


class TestServer(unittest.TestCase):
    
    def test_version(self):
        self.assertEqual(get_version(), "0.2.25")

    def test_server_init_del(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            server = Server()
            t = server.expr_type("forall (n m: Nat), n + m = m + n")
            del server
            server = Server()
            t = server.expr_type("forall (n m: Nat), n + m = m + n")
            del server
            server = Server()
            t = server.expr_type("forall (n m: Nat), n + m = m + n")
            del server

    def test_expr_type(self):
        server = Server()
        t = server.expr_type("forall (n m: Nat), n + m = m + n")
        self.assertEqual(t, "Prop")

    def test_goal_start(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[Variable(name="a", t="Prop")],
            target="∀ (q : Prop), a ∨ q → q ∨ a",
            name=None,
        )])
        self.assertEqual(str(state1.goals[0]),"a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a")

        del state0
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

        state0b = server.goal_start("forall (p: Prop), p -> p")
        del state0b
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

    def test_automatic_mode(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h", t="a ∨ b"),
            ],
            target="b ∨ a",
            name=None,
        )])
        state2 = server.goal_tactic(state1, goal_id=0, tactic="cases h")
        self.assertEqual(state2.goals, [
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            ),
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="b"),
                ],
                target="b ∨ a",
                name="inr",
            ),
        ])
        state3 = server.goal_tactic(state2, goal_id=1, tactic="apply Or.inl")
        state4 = server.goal_tactic(state3, goal_id=0, tactic="assumption")
        self.assertEqual(state4.goals, [
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            )
        ])

    def test_have(self):
        server = Server()
        state0 = server.goal_start("1 + 1 = 2")
        state1 = server.goal_tactic(state0, goal_id=0, tactic=TacticHave(branch="2 = 1 + 1", binder_name="h"))
        self.assertEqual(state1.goals, [
            Goal(
                variables=[],
                target="2 = 1 + 1",
            ),
            Goal(
                variables=[Variable(name="h", t="2 = 1 + 1")],
                target="1 + 1 = 2",
            ),
        ])
    def test_let(self):
        server = Server()
        state0 = server.goal_start("1 + 1 = 2")
        state1 = server.goal_tactic(
            state0, goal_id=0,
            tactic=TacticLet(branch="2 = 1 + 1", binder_name="h"))
        self.assertEqual(state1.goals, [
            Goal(
                variables=[],
                name="h",
                target="2 = 1 + 1",
            ),
            Goal(
                variables=[Variable(name="h", t="2 = 1 + 1", v="?h")],
                target="1 + 1 = 2",
            ),
        ])

    def test_conv_calc(self):
        server = Server(options={"automaticMode": False})
        state0 = server.goal_start("∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b")

        variables = [
            Variable(name="a", t="Nat"),
            Variable(name="b", t="Nat"),
            Variable(name="h", t="b = 2"),
        ]
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
        state2 = server.goal_tactic(state1, goal_id=0, tactic=TacticCalc("1 + a + 1 = a + 1 + 1"))
        self.assertEqual(state2.goals, [
            Goal(
                variables,
                target="1 + a + 1 = a + 1 + 1",
                name='calc',
            ),
            Goal(
                variables,
                target="a + 1 + 1 = a + b",
            ),
        ])
        state_c1 = server.goal_conv_begin(state2, goal_id=0)
        state_c2 = server.goal_tactic(state_c1, goal_id=0, tactic="rhs")
        state_c3 = server.goal_tactic(state_c2, goal_id=0, tactic="rw [Nat.add_comm]")
        state_c4 = server.goal_conv_end(state_c3)
        state_c5 = server.goal_tactic(state_c4, goal_id=0, tactic="rfl")
        self.assertTrue(state_c5.is_solved)

        state3 = server.goal_tactic(state2, goal_id=1, tactic=TacticCalc("_ = a + 2"))
        state4 = server.goal_tactic(state3, goal_id=0, tactic="rw [Nat.add_assoc]")
        self.assertTrue(state4.is_solved)

    def test_load_sorry(self):
        server = Server()
        unit, = server.load_sorry("example (p: Prop): p → p := sorry")
        self.assertIsNotNone(unit.goal_state, f"{unit.messages}")
        state0 = unit.goal_state
        self.assertEqual(state0.goals, [
            Goal(
                [Variable(name="p", t="Prop")],
                target="p → p",
            ),
        ])
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro h")
        state2 = server.goal_tactic(state1, goal_id=0, tactic="exact h")
        self.assertTrue(state2.is_solved)

        state1b = server.goal_tactic(state0, goal_id=0, tactic=TacticDraft("by\nhave h1 : Or p p := sorry\nsorry"))
        self.assertEqual(state1b.goals, [
            Goal(
                [Variable(name="p", t="Prop")],
                target="p ∨ p",
            ),
            Goal(
                [
                    Variable(name="p", t="Prop"),
                    Variable(name="h1", t="p ∨ p"),
                ],
                target="p → p",
            ),
        ])


    def test_env_add_inspect(self):
        server = Server()
        server.env_add(
            name="mystery",
            t="forall (n: Nat), Nat",
            v="fun (n: Nat) => n + 1",
            is_theorem=False,
        )
        inspect_result = server.env_inspect(name="mystery")
        self.assertEqual(inspect_result['type'], {'pp': 'Nat → Nat', 'dependentMVars': []})

    def test_goal_state_pickling(self):
        import tempfile
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        with tempfile.TemporaryDirectory() as td:
            path = td + "/goal-state.pickle"
            server.goal_save(state0, path)
            state0b = server.goal_load(path)
            self.assertEqual(state0b.goals, [
                Goal(
                    variables=[
                    ],
                    target="∀ (p q : Prop), p ∨ q → q ∨ p",
                )
            ])


if __name__ == '__main__':
    unittest.main()
