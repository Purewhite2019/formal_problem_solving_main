from abc import abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Set, Any
import collections as C
import heapq
import asyncio
import traceback

from loguru import logger
# logger = logger.opt(colors=True)  # Some tactics or proof states might has tokens like <...>, </...> that results in error in loguru.
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
from easydict import EasyDict
import vllm
from transformers import AutoTokenizer

from common.constants import BANNED_TOKENS, API_TIMEOUT, CODEBLOCK_PATTERN
from common.pantograph.server import Server, TacticFailure, ServerError
from common.pantograph.dataclasses import TacticHave, TacticLet, Tactic, GoalState, Goal, ProofSearchState, ProofSearchResult
from common.utils import zip_strict, remove_comments, normalize_spaces


class ProofSearchAgent:
    """
    A template best-first search agent for proof search. (The heuristic hasn't been implemented)
    """

    def __init__(self, max_search_trials: int) -> None:
        self.max_search_trials = max_search_trials

    @abstractmethod
    async def gen_tactics_async(
            self,
            state: ProofSearchState,
            goal_str: str,
        ) -> Tuple[List[Tactic], List[Any]]:
        """
        Given a GoalState, generate `self.num_samples_per_trial` tactics for its first goal, return parsed tactics and scores.
        """

    async def reset_async(self):
        """
        Clean garbabge
        """
        await logger.complete()

    @abstractmethod
    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Any) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
    
    async def search_async(self,
               server: Server,
               init_state: GoalState,
               tag: str='',
               verbose: bool=False,
               ignored_goals: List[Goal]=[] # Only works for goals that are not coupled w/ the first goal and are the last in the list 
               ) -> ProofSearchResult:
        """
        Best-first proof search from `init_state`
        """
        # Initialize
        assert server.is_automatic(), "Search must be run in automatic mode"
        time_start = time.time()
        
        if len(ignored_goals) > 0:
            try:
                for g in init_state.goals[:-len(ignored_goals)]:
                    assert all([i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep]), 'all([i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep])' # All coupled-goals should not be ignored
                for g_c, g_o in zip_strict(init_state.goals[-len(ignored_goals):], ignored_goals):
                    assert g_c.target == g_o.target, f'g_c.target == g_o.target'
                    assert g_c.name == g_o.name, f'g_c.name == g_o.name'
                    assert len(g_c.sibling_dep) == 0, 'len(g_c.sibling_dep) == 0'  # All coupled-goals should not be ignored
                    assert g_c.is_conversion == g_o.is_conversion, f'g_c.is_conversion == g_o.is_conversion'
                    assert all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)]), f'all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)])'
            except Exception as e:
                logger.error(f'Search({tag}): Error when handling ignored_goals: {e}\nState:\n{init_state.serialize()}\nIgnored goals:\n{ignored_goals}')
                raise RuntimeError(f'Search({tag}): Error when handling ignored_goals: {e}')
            del init_state.goals[-len(ignored_goals):]
        
        visited: Set[str] = set([
            str(init_state)
        ])
        search_pqueue: List[ProofSearchState] = [
            ProofSearchState(
                state=init_state,
                parent=None,
                last_step=None,
                cost=0.0
            )
        ]
        log = logger.info if verbose else logger.debug

        # Search
        try:
            for i_trial in range(self.max_search_trials):
                if len(search_pqueue) == 0:
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = 0, exiting.')
                    break
                else:
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = {len(search_pqueue)}')
                
                cur_search_state = heapq.heappop(search_pqueue)
                assert isinstance(cur_search_state, ProofSearchState)
                tactics, tac_infos = await self.gen_tactics_async(cur_search_state, cur_search_state.state.first_goal_str)  # By default using the first goal in a group of coupled metavariables

                for tactic, tac_info in zip_strict(tactics, tac_infos):
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.first_goal_str)}""" -> Tactic ```{str(tactic)}```')
                    try:
                        # TODO: Maybe recursive proof search can be used in `have` tactics?
                        cur_goal_id = cur_search_state.state.first_goal_id
                        next_goal_state = await server.goal_tactic_async(cur_search_state.state, cur_goal_id, tactic)  # Apply tactic at the first goal (since ALL goal should be solved, the order does not count.)

                        if len(ignored_goals) > 0:
                            try:
                                for g in next_goal_state.goals[:-len(ignored_goals)]:
                                    assert all([i < len(next_goal_state.goals) - len(ignored_goals) for i in g.sibling_dep]), 'all([i < len(next_goal_state.goals) - len(ignored_goals) for i in g.sibling_dep])' # All coupled-goals should not be ignored
                                for g_c, g_o in zip_strict(next_goal_state.goals[-len(ignored_goals):], ignored_goals):
                                    assert g_c.target == g_o.target, f'g_c.target == g_o.target'
                                    assert g_c.name == g_o.name, f'g_c.name == g_o.name'
                                    assert len(g_c.sibling_dep) == 0, 'len(g_c.sibling_dep) == 0'  # All coupled-goals should not be ignored
                                    assert g_c.is_conversion == g_o.is_conversion, f'g_c.is_conversion == g_o.is_conversion'
                                    assert all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)]), f'all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)])'
                            except Exception as e:
                                logger.error(f'Search({tag}): Error when handling ignored_goals: {e}\nState:\n{next_goal_state.serialize()}\nIgnored goals:\n{ignored_goals}')
                                raise RuntimeError(f'Search({tag}): Error when handling ignored_goals: {e}')
                            del next_goal_state.goals[-len(ignored_goals):]

                        # If proven, return
                        if next_goal_state.is_solved:
                            next_goal_state.goals.extend(ignored_goals) # If success, recover ignored goals
                            return ProofSearchResult(
                                duration=time.time() - time_start,
                                success=True,
                                proof=cur_search_state.tactic_history + [(cur_goal_id, tactic)],
                                final_state=next_goal_state,
                                states=cur_search_state.state_history
                            )
                        
                        # Otherwise, process new search state and continue search
                        next_search_state = ProofSearchState(
                            # state=next_goal_state,
                            state=GoalState(
                                state_id=next_goal_state.state_id,
                                goals=next_goal_state.goals,
                                payload={},
                                _sentinel=[]
                            ),
                            parent=cur_search_state,
                            last_step=(cur_goal_id, tactic),
                            cost=await self.guidance_async(cur_search_state, next_goal_state, tactic, tac_info)
                        )
                        if str(next_search_state) not in visited:
                            visited.add(str(next_search_state))
                            heapq.heappush(search_pqueue, next_search_state)

                    except TacticFailure as t:
                        log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.first_goal_str)}""", Tactic ```{str(tactic)}``` failed with ```{t}```')
                        # try the next tactic. this one failed
        except ServerError as e:
            logger.error(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.first_goal_str)}""", Tactic ```{str(tactic)}``` server failed with ```{e}```')
            # raise RuntimeError(f'While executing tactic: ```{tactic}``` at """{str(cur_search_state.state.first_goal_str)}"""') from e

        await self.reset_async()

        return ProofSearchResult(
            duration=time.time() - time_start,
            success=False,
        )

class DummyProofSearchAgent(ProofSearchAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(0)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    async def search_async(self,
               server: Server,
               init_state: GoalState,
               tag: str='',
               verbose: bool=False,
               ignored_goals: List[Goal]=[] # Only works for goals that are not coupled w/ the first goal and are the last in the list 
               ) -> ProofSearchResult:
        return ProofSearchResult(
            duration=0.0,
            success=False,
            proof=[],
            final_state=init_state,
            states=[init_state]
        )

class HammerProofSearchAgent(ProofSearchAgent):
    """
    A template best-first search agent for proof search. (The heuristic hasn't been implemented)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(1)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    async def gen_tactics_async(
            self,
            state: ProofSearchState,
            goal_str: str
        ) -> Tuple[List[Tactic], List[float]]:
        """
        Given a GoalState, generate `self.num_samples_per_trial` tactics for its first goal, return parsed tactics and scores.
        """
        return (['aesop'], [0.0])

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: float) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        return tac_info

class LLMProofSearchAgent(ProofSearchAgent):
    """
    A template best-first search agent for LLM-based proof search. (The heuristic hasn't been implemented)
    Multiple search agent (each for one proposition) + one AsyncOpenAI-style API server
    """

    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(max_search_trials)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_samples_per_trial = num_samples_per_trial
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """

    async def gen_tactics_async(
            self,
            state: ProofSearchState,
            goal_str: str
        ) -> Tuple[List[Tactic], List[Choice]]:
        """
        Given a GoalState, generate `self.num_samples_per_trial` tactics for its first goal, return parsed tactics and scores.
        """
        generated: Set[str] = set()
        tactics, tac_infos = [], []
        outputs: List[Choice] = []

        # Generate tactics
        try:
            if self.gen_model_name in ['deepseek-chat', 'deepseek-coder']:
                outputs = [r.choices[0] for r in asyncio.gather(*[
                    await self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=self.gen_prompt(state, goal_str),
                        max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                        stream=False,
                        temperature=self.temperature,
                        logprobs=True,
                        timeout=API_TIMEOUT
                    )
                    for _ in range(self.num_samples_per_trial)
                ])] # Deepseek API doesn't support `n`
            elif 'internlm' in self.gen_model_name.lower():
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(state, goal_str),
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                    stop='<|im_end|>',
                    timeout=API_TIMEOUT
                )).choices
            else:
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(state, goal_str),
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                    timeout=API_TIMEOUT
                )).choices
        except:
            logger.error(f'Failed to generate tactics:\n{traceback.format_exc()}')
        
        for output in outputs:
            # Neglect failed generations
            if not output.finish_reason == 'stop':
                continue
            # Remove replicates
            tactic = output.message.content.rstrip()
            normalized_tactic = normalize_spaces(remove_comments(tactic))
            
            if normalized_tactic not in generated:
                try:
                    generated.add(normalized_tactic)
                    # Parse tactics
                    # if 'sorry' in tactic:
                    #     if tactic.startswith('have'):
                    #         # The `have` tactic, equivalent to
                    #         # ```lean
                    #         # have {binder_name} : {branch} := sorry
                    #         # ```
                    #         decl, expr = tactic.split(':=')
                    #         assert 'sorry' in expr and 'sorry' not in decl, "'sorry' in expr and 'sorry' not in decl"

                    #         elems = decl.split()
                    #         assert elems[0] == 'have', "elems[0] == 'have'"
                    #         if elems[1] == ':':
                    #             binder_name = None
                    #             branch = ' '.join(elems[2:])
                    #         else:
                    #             assert elems[2] == ':', "elems[2] == ':'"
                    #             binder_name = elems[1]
                    #             branch = ' '.join(elems[3:])
                    #         tactic = TacticHave(
                    #             branch=branch,
                    #             binder_name=binder_name
                    #         )
                            
                    #     elif tactic.startswith('let'):
                    #         # The `let` tactic, equivalent to
                    #         # ```lean
                    #         # let {binder_name} : {branch} := sorry
                    #         # ```
                    #         decl, expr = tactic.split(':=')
                    #         assert 'sorry' in expr and 'sorry' not in decl, "'sorry' in expr and 'sorry' not in decl"

                    #         elems = decl.split()
                    #         assert elems[0] == 'let', "elems[0] == 'let'"
                    #         if elems[1] == ':':
                    #             binder_name = None
                    #             branch = ' '.join(elems[2:])
                    #         else:
                    #             assert elems[2] == ':', "elems[2] == ':'"
                    #             binder_name = elems[1]
                    #             branch = ' '.join(elems[3:])
                    #         tactic = TacticLet(
                    #             branch=branch,
                    #             binder_name=binder_name
                    #         )
                    #     logger.info(f'Parsed special tactic {tactic} from `{output.message.content}`')
                    #     # TODO: Support calc, conv and expr
                    # Detect banned token
                    assert all(banned_token not in normalized_tactic for banned_token in BANNED_TOKENS), 'Banned token detected'

                    tactics.append(tactic)
                    tac_infos.append(output)
                except Exception as e:
                    logger.debug(f'gen_tactics_async(): Tactic `{str(tactic)}` rejected due to {e}')
        return tactics, tac_infos

class StepProver_NALP_LLMProofSearchAgent(LLMProofSearchAgent):
    """
    A best-first search agent for proof search with normalized accumulated tactic log-probabilities as heuristic.
    Generator: InternLM2.5-StepProver
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    GEN_PROMPT_TEMPLATE = ("---\nNAME: {THEOREM_FULL_NAME}\n\n"
                        #    "---\nFILE:{theorem.file_path}\n\n"
                           "---\nPROOF_BEFORE: {PROOF_BEFORE}\n\n"
                           "---\nSTATE_BEFORE: {STATE}\n\n"
                           "---\nTACTIC: "
                        )

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        return [
            {
                "role": "user",
                "content": StepProver_NALP_LLMProofSearchAgent.GEN_PROMPT_TEMPLATE.replace('{THEOREM_FULL_NAME}', ' ').replace('{PROOF_BEFORE}', '\n'.join([str(t) for t in cur_search_state.tactic_history])).replace('{STATE}', str(goal_str))
            }
        ]

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)


class StepProver_Critic_LLMProofSearchAgent(StepProver_NALP_LLMProofSearchAgent):
    """
    A best-first search agent for proof search with a critic model as heuristic.
    Generator: InternLM2.5-StepProver
    Critic: InternLM2.5-StepProver
    """
    reward_token_id = 92527

    def format_chat(state: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": "Which state is closer to 'no goals'?"},
            {"role": "assistant", "content": state if state != '' else 'no goals'}
        ]

    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, critic_client: AsyncOpenAI, critic_model_name: str, *args, max_search_trials: int = 100, num_samples_per_trial: int = 32, temperature: Optional[float] = None, max_tokens: int = 256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

        self.critic_client = critic_client
        self.critic_tokenizer = AutoTokenizer.from_pretrained(critic_model_name, trust_remote_code=True)
        self.critic_model_name = critic_model_name
        self.reward_token = self.critic_tokenizer.decode([StepProver_Critic_LLMProofSearchAgent.reward_token_id])

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        chat = StepProver_Critic_LLMProofSearchAgent.format_chat('' if new_goal_state.is_solved else str(new_goal_state.first_goal_str))

        response = await self.critic_client.embeddings.create(
                model=self.critic_model_name,
                input=self.critic_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False),
                encoding_format='float'
        )
        value = response.data[0].embedding[-1]

        return -value

class SFT_NALP_LLMProofSearchAgent(LLMProofSearchAgent):
    """
    A best-first search agent for proof search with normalized accumulated tactic log-probabilities as heuristic.
    Generator: SFT
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        return [
            {
                "role": "system",
                "content": "You are a Lean 4 expert."
            },
            {
                "role": "user",
                "content": f"""Generate a tactic that can transform one step from the current tactic state to the 'no goals' tactic state.
Current tactic state:
```
{goal_str}
```
"""
            }
        ]

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)

class SFT_NALP_AVGGOAL_LLMProofSearchAgent(SFT_NALP_LLMProofSearchAgent):
    async def search_async(self,
               server: Server,
               init_state: GoalState,
               tag: str='',
               verbose: bool=False,
               ignored_goals: List[Goal]=[] # Only works for goals that are not coupled w/ the first goal and are the last in the list 
               ) -> ProofSearchResult:
        """
        Best-first proof search from `init_state`
        """
        # Initialize
        assert server.is_automatic(), "Search must be run in automatic mode"
        time_start = time.time()
        
        if len(ignored_goals) > 0:
            try:
                for g in init_state.goals[:-len(ignored_goals)]:
                    assert all([i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep]), 'all([i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep])' # All coupled-goals should not be ignored
                for g_c, g_o in zip_strict(init_state.goals[-len(ignored_goals):], ignored_goals):
                    assert g_c.target == g_o.target, f'g_c.target == g_o.target'
                    assert g_c.name == g_o.name, f'g_c.name == g_o.name'
                    assert len(g_c.sibling_dep) == 0, 'len(g_c.sibling_dep) == 0'  # All coupled-goals should not be ignored
                    assert g_c.is_conversion == g_o.is_conversion, f'g_c.is_conversion == g_o.is_conversion'
                    assert all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)]), f'all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)])'
            except Exception as e:
                logger.error(f'Search({tag}): Error when handling ignored_goals: {e}\nState:\n{init_state.serialize()}\nIgnored goals:\n{ignored_goals}')
                raise RuntimeError(f'Search({tag}): Error when handling ignored_goals: {e}')
            del init_state.goals[-len(ignored_goals):]
        
        visited: Set[str] = set([
            str(init_state)
        ])
        search_pqueue: List[ProofSearchState] = [
            ProofSearchState(
                state=init_state,
                parent=None,
                last_step=None,
                cost=0.0
            )
        ]
        log = logger.info if verbose else logger.debug

        # Search
        try:
            for i_trial in range(self.max_search_trials):
                if len(search_pqueue) == 0:
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = 0, exiting.')
                    break
                else:
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = {len(search_pqueue)}')
                
                cur_search_state = heapq.heappop(search_pqueue)
                assert isinstance(cur_search_state, ProofSearchState)
                
                # Average generate tactics for each coupled goals in the first coupled-group
                undirected_neighbors = C.defaultdict(list)
                for (i, g) in enumerate(cur_search_state.state.goals):
                    for neighbor in g.sibling_dep:
                        undirected_neighbors[i].append(neighbor)
                        undirected_neighbors[neighbor].append(i)

                visited = set()
                queue = C.deque([0])
                coupled_goal_ids = set([0])

                while len(queue) > 0:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        for neighbor in undirected_neighbors[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                                coupled_goal_ids.add(neighbor)

                coupled_goal_ids = sorted(coupled_goal_ids)
                coupled_goal_counts = [self.num_samples_per_trial // len(coupled_goal_ids) for _ in range(len(coupled_goal_ids))]
                for i in range((self.num_samples_per_trial % len(coupled_goal_ids))):
                    coupled_goal_counts[i] += 1
                assert sum(coupled_goal_counts) == self.num_samples_per_trial
                
                num_samples_per_trial = self.num_samples_per_trial
                goal_ids = []
                tactics = []
                tac_infos = []
                for (i, n) in zip_strict(coupled_goal_ids, coupled_goal_counts):
                    self.num_samples_per_trial = n
                    
                    cur_str = '\n'.join([
                        str(cur_search_state.state.goals[g]) for g in [i] + [j for j in coupled_goal_ids if j != i]
                    ])
                    tactics_i, tac_infos_i = await self.gen_tactics_async(cur_search_state, cur_str)  # By default using the first goal in a group of coupled metavariables
                    tactics += tactics_i
                    tac_infos += tac_infos_i
                    goal_ids += [i] * len(tactics_i)
                self.num_samples_per_trial = num_samples_per_trial

                for tactic, cur_goal_id, tac_info in zip_strict(tactics, goal_ids, tac_infos):
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.goals[cur_goal_id])}""" -> Tactic ```{str(tactic)}```')
                    try:
                        # TODO: Maybe recursive proof search can be used in `have` tactics?
                        next_goal_state = await server.goal_tactic_async(cur_search_state.state, cur_goal_id, tactic)  # Apply tactic at the first goal (since ALL goal should be solved, the order does not count.)

                        if len(ignored_goals) > 0:
                            try:
                                for g in next_goal_state.goals[:-len(ignored_goals)]:
                                    assert all([i < len(next_goal_state.goals) - len(ignored_goals) for i in g.sibling_dep]), 'all([i < len(next_goal_state.goals) - len(ignored_goals) for i in g.sibling_dep])' # All coupled-goals should not be ignored
                                for g_c, g_o in zip_strict(next_goal_state.goals[-len(ignored_goals):], ignored_goals):
                                    assert g_c.target == g_o.target, f'g_c.target == g_o.target'
                                    assert g_c.name == g_o.name, f'g_c.name == g_o.name'
                                    assert len(g_c.sibling_dep) == 0, 'len(g_c.sibling_dep) == 0'  # All coupled-goals should not be ignored
                                    assert g_c.is_conversion == g_o.is_conversion, f'g_c.is_conversion == g_o.is_conversion'
                                    assert all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)]), f'all([u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables)])'
                            except Exception as e:
                                logger.error(f'Search({tag}): Error when handling ignored_goals: {e}\nState:\n{next_goal_state.serialize()}\nIgnored goals:\n{ignored_goals}')
                                raise RuntimeError(f'Search({tag}): Error when handling ignored_goals: {e}')
                            del next_goal_state.goals[-len(ignored_goals):]

                        # If proven, return
                        if next_goal_state.is_solved:
                            next_goal_state.goals.extend(ignored_goals) # If success, recover ignored goals
                            return ProofSearchResult(
                                duration=time.time() - time_start,
                                success=True,
                                proof=cur_search_state.tactic_history + [(cur_goal_id, tactic)],
                                final_state=next_goal_state,
                                states=cur_search_state.state_history
                            )
                        
                        # Otherwise, process new search state and continue search
                        next_search_state = ProofSearchState(
                            # state=next_goal_state,
                            state=GoalState(
                                state_id=next_goal_state.state_id,
                                goals=next_goal_state.goals,
                                payload={},
                                _sentinel=[]
                            ),
                            parent=cur_search_state,
                            last_step=(cur_goal_id, tactic),
                            cost=await self.guidance_async(cur_search_state, next_goal_state, tactic, tac_info)
                        )
                        if str(next_search_state) not in visited:
                            visited.add(str(next_search_state))
                            heapq.heappush(search_pqueue, next_search_state)

                    except TacticFailure as e:
                        log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.goals[cur_goal_id])}""", Tactic ```{str(tactic)}``` failed with ```{repr(e)}```')
                    except ServerError as e:
                        log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Goal """{str(cur_search_state.state.goals[cur_goal_id])}""", Tactic ```{str(tactic)}``` failed with ```{repr(e)}```')
                        # try the next tactic. this one failed
        except Exception as e:
            logger.error(f'Search({tag}): {i_trial}/{self.max_search_trials}, failed with ```{repr(e)}```\n{traceback.format_exc()}')
            # raise RuntimeError(f'While executing tactic: ```{tactic}``` at """{str(cur_search_state.state.goals[cur_goal_id])}"""') from e

        await self.reset_async()

        return ProofSearchResult(
            duration=time.time() - time_start,
            success=False,
        )

class LeanSTaR_NALP_AVGGOAL_LLMProofSearchAgent(SFT_NALP_AVGGOAL_LLMProofSearchAgent):
    """
    A best-first search agent for proof search with normalized accumulated tactic log-probabilities as heuristic.
    Generator: Lean-STaR-plus
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        return [
            {
                "role": "user",
                "content": f"My LEAN 4 state is:\n```{goal_str}```\nPlease write down the reasoning that leads to the possible next tactic and then predict the tactic to help me prove the theorem."
            }
        ]

    async def gen_tactics_async(
            self,
            state: ProofSearchState,
            goal_str: str
        ) -> Tuple[List[Tactic], List[Choice]]:
        """
        Given a GoalState, generate `self.num_samples_per_trial` tactics for its first goal, return parsed tactics and scores.
        """
        generated: Set[str] = set()
        tactics, tac_infos = [], []
        outputs: List[Choice] = []

        # Generate tactics
        try:
            outputs = (await self.gen_client.chat.completions.create(
                model=self.gen_model_name,
                messages=self.gen_prompt(state, goal_str),
                max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                stream=False,
                temperature=self.temperature,
                n=self.num_samples_per_trial,
                logprobs=True,
                stop='<|im_end|>',
                timeout=API_TIMEOUT
            )).choices
        except:
            logger.error(f'Failed to generate tactics:\n{traceback.format_exc()}')
        
        for output in outputs:
            # Neglect failed generations
            if not output.finish_reason == 'stop':
                logger.warning('gen_tactics_async(): Max length exceeded.')
            # Remove replicates
            matches = CODEBLOCK_PATTERN.findall(output.message.content)
            if len(matches) == 0:
                logger.warning('gen_tactics_async(): no tactics matched in: ' + output.message.content)
                continue
            tactic = matches[0]
            normalized_tactic = normalize_spaces(remove_comments(tactic))
            
            if normalized_tactic not in generated:
                try:
                    generated.add(normalized_tactic)
                    assert all(banned_token not in normalized_tactic for banned_token in BANNED_TOKENS), 'Banned token detected'

                    tactics.append(tactic)
                    tac_infos.append(output)
                except Exception as e:
                    logger.debug(f'gen_tactics_async(): Tactic `{str(tactic)}` rejected due to {e}')
        return tactics, tac_infos

class StepProver_NALP_AVGGOAL_LLMProofSearchAgent(SFT_NALP_AVGGOAL_LLMProofSearchAgent):
    """
    A best-first search agent for proof search with normalized accumulated tactic log-probabilities as heuristic.
    Generator: InternLM2.5-StepProver
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    GEN_PROMPT_TEMPLATE = ("---\nNAME: {THEOREM_FULL_NAME}\n\n"
                        #    "---\nFILE:{theorem.file_path}\n\n"
                           "---\nPROOF_BEFORE: {PROOF_BEFORE}\n\n"
                           "---\nSTATE_BEFORE: {STATE}\n\n"
                           "---\nTACTIC: "
                        )

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator
        """
        return [
            {
                "role": "user",
                "content": StepProver_NALP_LLMProofSearchAgent.GEN_PROMPT_TEMPLATE.replace('{THEOREM_FULL_NAME}', ' ').replace('{PROOF_BEFORE}', '\n'.join([str(t) for t in cur_search_state.tactic_history])).replace('{STATE}', str(goal_str))
            }
        ]

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)


class StepProver_Critic_AVGGOAL_LLMProofSearchAgent(StepProver_NALP_AVGGOAL_LLMProofSearchAgent):
    """
    A best-first search agent for proof search with a critic model as heuristic.
    Generator: InternLM2.5-StepProver
    Critic: InternLM2.5-StepProver
    """
    reward_token_id = 92527

    def format_chat(state: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": "Which state is closer to 'no goals'?"},
            {"role": "assistant", "content": state if state != '' else 'no goals'}
        ]

    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, critic_client: AsyncOpenAI, critic_model_name: str, *args, max_search_trials: int = 100, num_samples_per_trial: int = 32, temperature: Optional[float] = None, max_tokens: int = 256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

        self.critic_client = critic_client
        self.critic_tokenizer = AutoTokenizer.from_pretrained(critic_model_name, trust_remote_code=True)
        self.critic_model_name = critic_model_name
        self.reward_token = self.critic_tokenizer.decode([StepProver_Critic_LLMProofSearchAgent.reward_token_id])

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
        chat = StepProver_Critic_LLMProofSearchAgent.format_chat('' if new_goal_state.is_solved else str(new_goal_state.first_goal_str))

        response = await self.critic_client.embeddings.create(
                model=self.critic_model_name,
                input=self.critic_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False),
                encoding_format='float'
        )
        value = response.data[0].embedding[-1]

        return -value
