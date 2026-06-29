from abc import abstractmethod
import asyncio
import collections as C
import heapq
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
import regex as re

from agent.base import ProofResult, ProverAgent
from common.constants import API_TIMEOUT, BANNED_TOKENS, CODEBLOCK_PATTERN
from common.pantograph.dataclasses import Goal, GoalState, ProofSearchState, Tactic
from common.pantograph.server import Server, ServerError, TacticFailure
from common.utils import normalize_spaces, remove_comments, zip_strict


class ProofSearchAgent(ProverAgent):
    agent_name = "proof_search"

    def __init__(self, max_search_trials: int) -> None:
        self.max_search_trials = max_search_trials

    @abstractmethod
    async def gen_tactics_async(self, state: ProofSearchState, goal_str: str) -> Tuple[List[Tactic], List[Any]]:
        pass

    async def reset_async(self):
        await logger.complete()

    @abstractmethod
    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Any) -> float:
        pass

    async def prove_async(
        self,
        sample,
        server,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
    ) -> ProofResult:
        return await self.search_async(server.server if hasattr(server, "server") else server, init_state, tag, verbose)

    async def search_async(
        self,
        server: Server,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
        ignored_goals: List[Goal] = [],
    ) -> ProofResult:
        assert server.is_automatic(), "Search must be run in automatic mode"
        time_start = time.time()

        if len(ignored_goals) > 0:
            try:
                for g in init_state.goals[:-len(ignored_goals)]:
                    assert all(i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep)
                for g_c, g_o in zip_strict(init_state.goals[-len(ignored_goals):], ignored_goals):
                    assert g_c.target == g_o.target
                    assert g_c.name == g_o.name
                    assert len(g_c.sibling_dep) == 0
                    assert g_c.is_conversion == g_o.is_conversion
                    assert all(u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables))
            except Exception as e:
                logger.error(f"Search({tag}): Error when handling ignored_goals: {e}\nState:\n{init_state.serialize()}\nIgnored goals:\n{ignored_goals}")
                raise RuntimeError(f"Search({tag}): Error when handling ignored_goals: {e}")
            del init_state.goals[-len(ignored_goals):]

        visited: Set[str] = {str(init_state)}
        search_pqueue: List[ProofSearchState] = [ProofSearchState(state=init_state, parent=None, last_step=None, cost=0.0)]
        log = logger.info if verbose else logger.debug

        try:
            for i_trial in range(self.max_search_trials):
                if len(search_pqueue) == 0:
                    log(f"Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = 0, exiting.")
                    break
                log(f"Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = {len(search_pqueue)}")

                cur_search_state = heapq.heappop(search_pqueue)
                tactics, tac_infos = await self.gen_tactics_async(cur_search_state, cur_search_state.state.first_goal_str)

                for tactic, tac_info in zip_strict(tactics, tac_infos):
                    log(f"Search({tag}): {i_trial}/{self.max_search_trials}, Goal \"\"\"{str(cur_search_state.state.first_goal_str)}\"\"\" -> Tactic ```{str(tactic)}```")
                    try:
                        cur_goal_id = cur_search_state.state.first_goal_id
                        next_goal_state = await server.goal_tactic_async(cur_search_state.state, cur_goal_id, tactic)

                        if len(ignored_goals) > 0:
                            del next_goal_state.goals[-len(ignored_goals):]

                        if next_goal_state.is_solved:
                            next_goal_state.goals.extend(ignored_goals)
                            proof = cur_search_state.tactic_history + [(cur_goal_id, tactic)]
                            return ProofResult(
                                duration=time.time() - time_start,
                                success=True,
                                proof=proof,
                                final_state=next_goal_state,
                                metadata={"states": [s.serialize() for s in cur_search_state.state_history]},
                            )

                        next_search_state = ProofSearchState(
                            state=GoalState(state_id=next_goal_state.state_id, goals=next_goal_state.goals, payload={}, _sentinel=[]),
                            parent=cur_search_state,
                            last_step=(cur_goal_id, tactic),
                            cost=await self.guidance_async(cur_search_state, next_goal_state, tactic, tac_info),
                        )
                        if str(next_search_state) not in visited:
                            visited.add(str(next_search_state))
                            heapq.heappush(search_pqueue, next_search_state)
                    except TacticFailure as t:
                        log(f"Search({tag}): tactic ```{str(tactic)}``` failed with ```{t}```")
        except ServerError as e:
            logger.error(f"Search({tag}) server failed with ```{e}```")

        await self.reset_async()
        return ProofResult(duration=time.time() - time_start, success=False)


class DummyProofSearchAgent(ProofSearchAgent):
    agent_name = "dummy"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(0)

    async def prove_async(self, sample, server, init_state: GoalState, tag: str = "", verbose: bool = False) -> ProofResult:
        return ProofResult(duration=0.0, success=False, proof=[], final_state=init_state)

    async def gen_tactics_async(self, state: ProofSearchState, goal_str: str) -> Tuple[List[Tactic], List[Any]]:
        return [], []

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Any) -> float:
        return 0.0


class LLMProofSearchAgent(ProofSearchAgent):
    def __init__(
        self,
        gen_client: AsyncOpenAI,
        gen_model_name: str,
        *args,
        max_search_trials: int = 100,
        num_samples_per_trial: int = 32,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(max_search_trials)
        if args or kwargs:
            logger.warning(f"Redundant arguments for {type(self)}: {args} {kwargs}")
        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_samples_per_trial = num_samples_per_trial
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        pass

    async def gen_tactics_async(self, state: ProofSearchState, goal_str: str) -> Tuple[List[Tactic], List[Choice]]:
        generated: Set[str] = set()
        tactics, tac_infos = [], []
        outputs: List[Choice] = []

        try:
            if self.gen_model_name in ["deepseek-chat", "deepseek-coder"]:
                responses = await asyncio.gather(*[
                    self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=self.gen_prompt(state, goal_str),
                        max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                        stream=False,
                        temperature=self.temperature,
                        logprobs=True,
                        timeout=API_TIMEOUT,
                    )
                    for _ in range(self.num_samples_per_trial)
                ])
                outputs = [r.choices[0] for r in responses]
            elif "internlm" in self.gen_model_name.lower():
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(state, goal_str),
                    max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                    stop=["<|im_end|>", "\n\n"],
                    timeout=API_TIMEOUT,
                )).choices
            else:
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(state, goal_str),
                    max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                    timeout=API_TIMEOUT,
                )).choices
        except Exception:
            logger.error(f"Failed to generate tactics:\n{traceback.format_exc()}")

        for output in outputs:
            if not output.finish_reason == "stop":
                continue
            tactic = output.message.content.rstrip()
            normalized_tactic = normalize_spaces(remove_comments(tactic))

            if normalized_tactic not in generated:
                try:
                    generated.add(normalized_tactic)
                    assert all(banned_token not in normalized_tactic for banned_token in BANNED_TOKENS), "Banned token detected"
                    tactics.append(tactic)
                    tac_infos.append(output)
                except Exception as e:
                    logger.debug(f"gen_tactics_async(): Tactic `{str(tactic)}` rejected due to {e}")
        return tactics, tac_infos

    async def _search_async_avggoal(
        self,
        server: Server,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
        ignored_goals: List[Goal] = [],
    ) -> ProofResult:
        assert server.is_automatic(), "Search must be run in automatic mode"
        time_start = time.time()

        if len(ignored_goals) > 0:
            try:
                for g in init_state.goals[:-len(ignored_goals)]:
                    assert all(i < len(init_state.goals) - len(ignored_goals) for i in g.sibling_dep)
                for g_c, g_o in zip_strict(init_state.goals[-len(ignored_goals):], ignored_goals):
                    assert g_c.target == g_o.target
                    assert g_c.name == g_o.name
                    assert len(g_c.sibling_dep) == 0
                    assert g_c.is_conversion == g_o.is_conversion
                    assert all(u.name == v.name for u, v in zip_strict(g_c.variables, g_o.variables))
            except Exception as e:
                logger.error(f"Search({tag}): Error when handling ignored_goals: {e}")
                raise RuntimeError(f"Search({tag}): Error when handling ignored_goals: {e}")
            del init_state.goals[-len(ignored_goals):]

        visited: Set[str] = {str(init_state)}
        search_pqueue: List[ProofSearchState] = [ProofSearchState(
            state=init_state, parent=None, last_step=None, cost=0.0)]
        log = logger.info if verbose else logger.debug

        try:
            for i_trial in range(self.max_search_trials):
                if len(search_pqueue) == 0:
                    log(f"Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = 0, exiting.")
                    break
                log(f"Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = {len(search_pqueue)}")

                cur_search_state = heapq.heappop(search_pqueue)
                assert isinstance(cur_search_state, ProofSearchState)

                undirected_neighbors = C.defaultdict(list)
                for (i, g) in enumerate(cur_search_state.state.goals):
                    for neighbor in g.sibling_dep:
                        undirected_neighbors[i].append(neighbor)
                        undirected_neighbors[neighbor].append(i)

                visited_local: Set[int] = set()
                queue = C.deque([0])
                coupled_goal_ids: set = {0}
                while len(queue) > 0:
                    current = queue.popleft()
                    if current not in visited_local:
                        visited_local.add(current)
                        for neighbor in undirected_neighbors[current]:
                            if neighbor not in visited_local:
                                queue.append(neighbor)
                                coupled_goal_ids.add(neighbor)

                coupled_goal_ids = sorted(coupled_goal_ids)
                coupled_goal_counts = [
                    self.num_samples_per_trial // len(coupled_goal_ids)
                    for _ in range(len(coupled_goal_ids))]
                for i in range(self.num_samples_per_trial % len(coupled_goal_ids)):
                    coupled_goal_counts[i] += 1
                assert sum(coupled_goal_counts) == self.num_samples_per_trial

                num_samples_saved = self.num_samples_per_trial
                goal_ids: list = []
                all_tactics: list = []
                all_tac_infos: list = []
                for (goal_i, n) in zip_strict(coupled_goal_ids, coupled_goal_counts):
                    self.num_samples_per_trial = n
                    cur_str = "\n".join([
                        str(cur_search_state.state.goals[g])
                        for g in [goal_i] + [j for j in coupled_goal_ids if j != goal_i]])
                    tactics_i, tac_infos_i = await self.gen_tactics_async(
                        cur_search_state, cur_str)
                    all_tactics += tactics_i
                    all_tac_infos += tac_infos_i
                    goal_ids += [goal_i] * len(tactics_i)
                self.num_samples_per_trial = num_samples_saved

                for tactic, cur_goal_id, tac_info in zip_strict(all_tactics, goal_ids, all_tac_infos):
                    log(f"Search({tag}): {i_trial}/{self.max_search_trials}, "
                        f"Goal \"\"\"{str(cur_search_state.state.goals[cur_goal_id])}\"\"\""
                        f" -> Tactic ```{str(tactic)}```")
                    try:
                        next_goal_state = await server.goal_tactic_async(
                            cur_search_state.state, cur_goal_id, tactic)

                        if len(ignored_goals) > 0:
                            del next_goal_state.goals[-len(ignored_goals):]

                        if next_goal_state.is_solved:
                            next_goal_state.goals.extend(ignored_goals)
                            proof = cur_search_state.tactic_history + [(cur_goal_id, tactic)]
                            return ProofResult(
                                duration=time.time() - time_start,
                                success=True,
                                proof=proof,
                                final_state=next_goal_state,
                                metadata={"states": [s.serialize() for s in cur_search_state.state_history]},
                            )

                        next_search_state = ProofSearchState(
                            state=GoalState(state_id=next_goal_state.state_id, goals=next_goal_state.goals, payload={}, _sentinel=[]),
                            parent=cur_search_state,
                            last_step=(cur_goal_id, tactic),
                            cost=await self.guidance_async(cur_search_state, next_goal_state, tactic, tac_info),
                        )
                        if str(next_search_state) not in visited:
                            visited.add(str(next_search_state))
                            heapq.heappush(search_pqueue, next_search_state)
                    except TacticFailure as t:
                        log(f"Search({tag}): tactic ```{str(tactic)}``` failed with ```{t}```")
        except ServerError as e:
            logger.error(f"Search({tag}) server failed with ```{e}```")

        await self.reset_async()
        return ProofResult(duration=time.time() - time_start, success=False)


class StepProverAgent(LLMProofSearchAgent):
    agent_name = "stepprover"
    GEN_PROMPT_TEMPLATE = (
        "---\nNAME: {THEOREM_FULL_NAME}\n\n"
        "---\nPROOF_BEFORE: {PROOF_BEFORE}\n\n"
        "---\nSTATE_BEFORE: {STATE}\n\n"
        "---\nTACTIC: "
    )

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        return [{
            "role": "user",
            "content": self.GEN_PROMPT_TEMPLATE
                .replace("{THEOREM_FULL_NAME}", " ")
                .replace("{PROOF_BEFORE}", "\n".join([str(t) for t in cur_search_state.tactic_history]))
                .replace("{STATE}", str(goal_str)),
        }]

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)

    async def search_async(
        self, server: Server, init_state: GoalState, tag: str = "",
        verbose: bool = False, ignored_goals: List[Goal] = [],
    ) -> ProofResult:
        return await self._search_async_avggoal(server, init_state, tag, verbose, ignored_goals)


class LeanSTaRAgent(LLMProofSearchAgent):
    agent_name = "leanstar"

    def gen_prompt(self, cur_search_state: ProofSearchState, goal_str: str) -> List[Dict[str, str]]:
        return [{
            "role": "user",
            "content": f"My LEAN 4 state is:\n```{goal_str}```\nPlease write down the reasoning that leads to the possible next tactic and then predict the tactic to help me prove the theorem.",
        }]

    async def gen_tactics_async(self, state: ProofSearchState, goal_str: str) -> Tuple[List[Tactic], List[Choice]]:
        generated: Set[str] = set()
        tactics, tac_infos = [], []
        outputs: List[Choice] = []

        try:
            outputs = (await self.gen_client.chat.completions.create(
                model=self.gen_model_name,
                messages=self.gen_prompt(state, goal_str),
                max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                stream=False,
                temperature=self.temperature,
                n=self.num_samples_per_trial,
                logprobs=True,
                stop=["<|im_end|>"],
                timeout=API_TIMEOUT,
            )).choices
        except Exception:
            logger.error(f"Failed to generate tactics:\n{traceback.format_exc()}")

        for output in outputs:
            if output.finish_reason != "stop":
                logger.debug(f"gen_tactics_async(): Max length exceeded. output.message.content={[output.message.content]}")

            text = output.message.content
            parse_result = re.findall(CODEBLOCK_PATTERN, text)
            if len(parse_result) > 0:
                tactic = parse_result[0].strip()
            else:
                split_cnt = len(re.findall("```", text))
                if split_cnt == 0:
                    tactic = text.strip()
                elif split_cnt == 1:
                    tactic = text[:text.find("```")].strip()
                else:
                    tactic = text

            normalized_tactic = normalize_spaces(remove_comments(tactic))
            if normalized_tactic not in generated:
                try:
                    generated.add(normalized_tactic)
                    assert all(banned_token not in normalized_tactic for banned_token in BANNED_TOKENS), "Banned token detected"
                    tactics.append(tactic)
                    tac_infos.append(output)
                except Exception as e:
                    logger.debug(f"gen_tactics_async(): Tactic `{str(tactic)}` rejected due to {e}")
        return tactics, tac_infos

    async def guidance_async(self, cur_search_state: ProofSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)

    async def search_async(
        self, server: Server, init_state: GoalState, tag: str = "",
        verbose: bool = False, ignored_goals: List[Goal] = [],
    ) -> ProofResult:
        return await self._search_async_avggoal(server, init_state, tag, verbose, ignored_goals)
