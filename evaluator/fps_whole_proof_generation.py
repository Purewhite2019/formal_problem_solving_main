import os
import os.path as osp
import sys
import subprocess
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path
from typing import Dict, Set, List
import time
import pickle
import regex as re

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
import openai
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
import argparse
from tqdm import tqdm
from termcolor import colored
from loguru import logger
from fire import Fire

from common.constants import CORE_OPTIONS, BANNED_TOKENS, PROBLEM_KEYS
from common.pantograph.dataclasses import FormalProblem, ProofSearchResult
from common.pantograph.solving_server import TermSolvingServer
from agent.proof_search import ProofSearchResult
from common.utils import remove_comments


LEAN4_DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
OPTION_HEADER = '\n'.join(f'set_option ' + o.replace('=', ' ') + ' in' for o in CORE_OPTIONS)


def deepseek_prover_format_prompt(data: FormalProblem, formal_statement: str, tac_history: List[str]) -> str:
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=LEAN4_DEFAULT_IMPORTS + '\n\n',  # We don't add `open`s because tactic mode doesn't support `open`
        informal_prefix=f"/-- {data.informal_problem}-/\n",
        formal_statement=formal_statement + 'by\n' + '\n'.join([t + ';' for t in tac_history]) + '\n',
    )

def deepseek_prover_post_process(output: str) -> str:
    # Truncate at '````
    _find_idx = output.find("```")
    code = output[:_find_idx] if _find_idx >= 0 else output
    
    # Remove starting spaces
    lines = code.splitlines()
    min_spaces = float('inf')
    for line in lines:
        space_count = len(line) - len(line.lstrip(' '))
        
        if space_count < min_spaces and line.strip():
            min_spaces = space_count
    
    if min_spaces == float('inf'):
        min_spaces = 0
    
    code = '\n'.join([l[min_spaces:] for l in lines])
    return code

def theoremllama_format_prompt(data: FormalProblem, formal_statement: str, tac_history: List[str]) -> str:
    return f'''Natural language version of theorem and proof:
{data.informal_problem}
Show that it is {data.informal_answer}.

{data.informal_solution}

### Lean4 version of theorem statement:
```lean
{formal_statement}
```

### Lean4 version of theorem and proof:
```lean
{formal_statement} by
''' + '\n'.join([t + ';' for t in tac_history]) + '\n'

def theoremllama_post_process(output: str) -> str:
    # Truncate at '````
    _find_idx = output.find("```")
    code = output[:_find_idx] if _find_idx >= 0 else output
    
    # Remove starting spaces
    lines = code.splitlines()
    min_spaces = float('inf')
    for line in lines:
        space_count = len(line) - len(line.lstrip(' '))
        
        if space_count < min_spaces and line.strip():
            min_spaces = space_count
    
    if min_spaces == float('inf'):
        min_spaces = 0
    
    code = '\n'.join([l[min_spaces:] for l in lines])
    return code

TEMPLATE_DICT = {
    'deepseek_prover' : (deepseek_prover_format_prompt, deepseek_prover_post_process),
    'theoremllama' : (theoremllama_format_prompt, theoremllama_post_process)
}


def main(
        log_root: str,
        model: str,
        benchmark: str,
        benchmark_root: str='data/',
        project_root: str='data/mathlib4/',
        num_samples_per_trial: int=128,
        temperature: float=1.0,
        top_p: float=0.95,
        max_tokens: int=2048,
        num_concurrency: int=12,
        verbose: bool=False,
        dry_run: bool=False,
        gen_base_url: Optional[str]='',
        gen_api_key: Optional[str]='',
        gen_model_name: Optional[str]='',
        only_proving: bool=False,
        resume_from: Optional[str]=None,
):
    saved_args = {**locals()}
    
    model = model.lower()
    benchmark = benchmark.lower()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = ('proving.' if only_proving else 'solving.')+benchmark+'.'+'whole_proof_generation'+'.'
    
    assert model in TEMPLATE_DICT.keys()
    format_prompt, post_process = TEMPLATE_DICT[model]

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Running {benchmark} proof search experiment with hyperparams: {saved_args}')
    if only_proving:
        logger.warning('Running on proving mode.')
    log_debug = logger.info if verbose else logger.debug

    gen_client = AsyncOpenAI(
        base_url=gen_base_url,
        api_key=gen_api_key
    )

    # Load data
    samples = []
    with open(osp.join(benchmark_root, benchmark+'.jsonl'), 'r') as f:
        samples = [FormalProblem(**json.loads(l)) for l in f.readlines()]
    
    finished = []
    logger.info(f"Loaded {len(samples)} samples for {benchmark} from {osp.join(benchmark_root, benchmark+'.jsonl')}")
    
    # Resume from interrupted experiments
    if resume_from is not None:
        load_file = [p for p in os.listdir(resume_from) if p.startswith(log_prefix) and p.endswith('.pkl')]
        if len(load_file) != 1:
            logger.error(f'There should be only one loadable experiment log in "{resume_from}", but detected {load_file}')
            return
        load_file = load_file[0]
        with open(osp.join(resume_from, load_file), 'rb') as f:
            finished = pickle.load(f)
        samples_str_all = set([
            (d.informal_problem, d.informal_answer, d.informal_solution, d.formal_problem, d.formal_answer) for d in samples
        ])
        finished_str_all = set([
            (d['informal_problem'], d['informal_answer'], d['informal_solution'], d['formal_problem'], d['formal_answer']) for d in finished
        ])
        assert len(samples_str_all) == len(samples)
        assert len(finished_str_all) == len(finished)
        assert finished_str_all.issubset(samples_str_all)
        samples = [d for d in samples if (d.informal_problem, d.informal_answer, d.informal_solution, d.formal_problem, d.formal_answer) not in finished_str_all]
        logger.critical(f'Resumed {len(finished)} samples from {osp.join(resume_from, load_file)}, now remaining {len(samples)} samples to evaluate.')

    async def search(sample: FormalProblem, tag_i: int) -> None:
        n_prompt_tokens = 0
        n_completion_tokens = 0
        try:
            server = TermSolvingServer(
                imports=["Mathlib", "Aesop"],
                project_path=project_root,
                timeout=300,
                tag=str(tag_i)
            )
            log_debug(f"search({tag_i}): server initialized.")
            if only_proving:
                init_solution_state = await server.init_backward_state_async(sample)
                tac_history = []
            else:
                init_solution_state, tac_history = await server.init_solving_state_async(sample)
            assert server.formal_problem_framework.endswith(':= sorry\n')
            
            logger.info(f"search({tag_i}): initial state ```\n{str(init_solution_state)}\n```.")
            if dry_run:
                return
            
            start_time = time.time()
            prompt = format_prompt(sample, server.formal_problem_framework[:-len('sorry\n')], tac_history)
            cur_solution_state = None
            for i in range(num_samples_per_trial):
                try:
                    response : openai.types.completion.Completion = (await gen_client.completions.create(
                        model=gen_model_name,
                        prompt=prompt,
                        max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN),
                        stream=False,
                        temperature=temperature,
                        top_p=top_p,
                        n=1,
                        stop='```'
                    ))
                    n_prompt_tokens += response.usage.prompt_tokens
                    n_completion_tokens += response.usage.completion_tokens
                    proof = '{\n' + remove_comments(post_process(response.choices[0].text)) + '\n}'
                    assert all(banned_token not in proof for banned_token in BANNED_TOKENS), f'Banned token detected in proof:{proof}'
                    
                    cur_solution_state = await server.server.goal_tactic_async(init_solution_state, 0, proof)
                    assert cur_solution_state.is_solved, f'Solution state is not solved\n{cur_solution_state}'
                    break
                except Exception as e:
                    log_debug(f'search({tag_i}, {i}/{num_samples_per_trial}): failed due to {repr(e)}')
            log_debug(f'search({tag_i}): Token usage: n_prompt_tokens={n_prompt_tokens}, n_completion_tokens={n_completion_tokens}')

            if cur_solution_state is not None and cur_solution_state.is_solved:
                search_result = ProofSearchResult(
                    duration=time.time() - start_time,
                    success=True,
                    proof=[(0, proof)]
                )
            else:
                search_result = ProofSearchResult(
                    duration=time.time() - start_time,
                    success=False,
                )
            
            sample = sample.__dict__.copy()
            for k in set(sample.keys()).difference(PROBLEM_KEYS):
                sample.pop(k)
            sample['search_result'] = search_result
            
            logger.opt(colors=True).info(f"search({tag_i}): " + ('<green>Proof search succeeded</green>' if search_result.success else '<yellow>failed</yellow>' + f' in {search_result.duration} (s)'))
            if search_result.success:
                logger.info(f'search({tag_i}): \n{str(init_solution_state)}\n{proof}')
                if not only_proving:
                    try:
                        submission = await server.get_submission_async(cur_solution_state)
                        sample['submission'] = submission
                        eq_proof = await server.check_rpe_async(submission)
                        sample['eq_proof'] = eq_proof
                        logger.opt(colors=True).info(f"search({tag_i}): " + ('<green>Solution eq succeeded</green>' if eq_proof is not None else '<yellow>failed</yellow>'))
                        logger.info(f'search({tag_i}): submission: {submission}, eq_proof: {eq_proof}')
                    except Exception as e:
                        logger.error(f'search({tag_i}): failed to evaluate direct answer due to {repr(e)}:\n{traceback.format_exc()}')
            finished.append(sample)
        except:
            logger.error(f"search({tag_i}): failed because {traceback.format_exc()}")

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, sample in tqdm(enumerate(samples)):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    search(sample, i)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished search, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
            with open(osp.join(log_root, log_prefix+now+'.jsonl'), 'w') as f:
                for sample in finished:
                    sample['search_result'] = sample['search_result'].serialize()
                    f.write(json.dumps(sample)+'\n')
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
