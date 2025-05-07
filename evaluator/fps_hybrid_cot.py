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
from openai import OpenAI, AsyncOpenAI, RateLimitError, NOT_GIVEN
import argparse
from tqdm import tqdm
from termcolor import colored
from loguru import logger
from fire import Fire

from common.constants import CORE_OPTIONS, BANNED_TOKENS, CODEBLOCK_PATTERN, RETRY_WAIT_TIME, PROBLEM_KEYS
from common.pantograph.dataclasses import ProofSearchResult, FormalProblem
from common.pantograph.server import ServerError, TacticFailure
from common.pantograph.solving_server import PropSolvingServer
from agent.proof_search import ProofSearchResult
from common.utils import remove_comments


LEAN4_DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
OPTION_HEADER = '\n'.join(f'set_option ' + o.replace('=', ' ') + ' in' for o in CORE_OPTIONS)
N_API_CHUNK = 1

def format_sample(informal_problem: str, forward_context: str, formal_solution: Optional[str]=None) -> str:
    assert informal_problem is not None
    assert forward_context is not None
    text = f'''# Informal Problem
"""
{informal_problem}
"""
# Goal State
```lean4
{forward_context}
```
'''
    if formal_solution is not None:
        text += f'''# Formal Proof
```lean4
{formal_solution}
```
'''
    return text

def format_hybrid_cot_prompt(demonstrations: List[Dict], informal_problem: str, forward_context: str) -> str:
    prompt = '''Given an informal math problem and a corresponding Lean 4 goal state, please think step by step and construct a formal proof deducing the answer.
Please assume the following header code has already been executed, and do not add any imports or openings.
```lean4
import Mathlib
import Aesop
```

Here are some examples:

'''
    for v in demonstrations:
        assert v.get('formal_solution', None) is not None
        prompt += format_sample(v['informal_problem'], v['forward_context'], v['formal_solution'])
        prompt += '\n---\n\n'

    prompt += 'Now, please generate a formal proof for the following problem.\n\n'
    
    prompt += format_sample(informal_problem, forward_context) + '\n'
    return prompt

def post_process(output: str) -> str:
    try:
        code = re.findall(CODEBLOCK_PATTERN, output)[0]
    except IndexError:
        code = output
    
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


def main(
        log_root: str,
        benchmark: str,
        demonstration_path: str='data/prompt_hybrid_cot_final.jsonl',
        benchmark_root: str='data/',
        project_root: str='data/mathlib4/',
        num_samples_per_trial: int=16,
        temperature: float=1.0,
        top_p: float=0.95,
        max_tokens: int=2048,
        num_concurrency: int=24,
        verbose: bool=False,
        dry_run: bool=False,
        gen_base_url: Optional[str]=None,
        gen_api_key: Optional[str]=None,
        gen_model_name: Optional[str]=None,
        resume_from: Optional[str]=None,
):
    saved_args = {**locals()}
    
    benchmark = benchmark.lower()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = benchmark+'.'+'hybrid_cot'+'.'
    assert num_samples_per_trial % N_API_CHUNK == 0

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Running {benchmark} proof search experiment with hyperparams: {saved_args}')
    log_debug = logger.info if verbose else logger.debug

    gen_client = AsyncOpenAI(
        base_url=gen_base_url,
        api_key=gen_api_key
    )

    # Load data
    with open(demonstration_path, 'r') as f:
        demonstrations = [json.loads(l) for l in f.readlines()]
        logger.info(f"Loaded {len(demonstrations)} demonstrations from {demonstration_path}")
    
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
            server = PropSolvingServer(
                imports=["Mathlib", "Aesop"],
                project_path=project_root,
                timeout=300,
                tag=str(tag_i)
            )
            log_debug(f"search({tag_i}): server initialized.")
            init_solution_state = await server.init_solving_state_async(sample)
                
            forward_context = str(init_solution_state.goals[0])
            assert forward_context.startswith('case h.mp\n') and forward_context.endswith('\n⊢ ?w')
            forward_context = forward_context[len('case h.mp\n'):]
            
            log_debug(f"search({tag_i}): initial state ```\n{forward_context}\n```.")
            if dry_run:
                return
            
            start_time = time.time()
            prompt = format_hybrid_cot_prompt(demonstrations, sample.informal_problem, forward_context)
            n_api_retry = 0
            
            sample = sample.__dict__.copy()
            for k in set(sample.keys()).difference(PROBLEM_KEYS):
                sample.pop(k)
            sample['responses'] = []
            is_success = False
            for i in range(num_samples_per_trial // N_API_CHUNK):
                while True:
                    try:
                        responses : openai.types.completion.Completion = (await gen_client.chat.completions.create(
                            model=gen_model_name,
                            messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a helpful assistant."
                                        },
                                        {
                                            "role": "user",
                                            "content": prompt
                                        }
                                    ],
                            max_tokens=(max_tokens if max_tokens > 0 else NOT_GIVEN),
                            stream=False,
                            temperature=temperature,
                            top_p=top_p,
                            n=N_API_CHUNK
                        ))
                        break
                    except (json.decoder.JSONDecodeError, openai.RateLimitError) as e:
                        await asyncio.sleep(RETRY_WAIT_TIME)
                        n_api_retry += 1
                        log_debug(f'search({tag_i}, {i*N_API_CHUNK}/{num_samples_per_trial}): {n_api_retry}-th try due to {repr(e)}')
                n_prompt_tokens += 0 if responses.usage is None else responses.usage.prompt_tokens
                n_completion_tokens += 0 if responses.usage is None else responses.usage.completion_tokens
                sample['responses'].append(responses)
                
                for j, choice in enumerate(responses.choices):
                    try:
                        proof = '{\n' + remove_comments(post_process(choice.message.content)) + '\n}'
                        if any(banned_token in proof for banned_token in BANNED_TOKENS):
                            log_debug(f'search({tag_i}, {i*N_API_CHUNK+j}/{num_samples_per_trial}): Banned token detected in proof:{proof}')
                            continue
                        
                        await server.server.restart_async()
                        cur_solution_state = await server.init_solving_state_async()
                        cur_solution_state = await server.server.goal_tactic_async(cur_solution_state, 0, proof)
                        assert [g.name for g in cur_solution_state.goals] == ['h.mpr'], f'Forward solution state is not solved\n{cur_solution_state}'
                        is_success = True
                        break
                    except (TacticFailure, ServerError) as e:
                        log_debug(f'search({tag_i}, {i*N_API_CHUNK+j}/{num_samples_per_trial}): failed due to {repr(e)}')
                    except Exception as e:
                        logger.info(f'search({tag_i}, {i*N_API_CHUNK+j}/{num_samples_per_trial}): failed due to {repr(e)}\n{traceback.format_exc()}')
                
                if is_success:
                    break
            
            logger.info(f'search({tag_i}): Token usage: n_prompt_tokens={n_prompt_tokens}, n_completion_tokens={n_completion_tokens}')

            if is_success:
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
            sample['search_result'] = search_result
            logger.opt(colors=True).info(f"search({tag_i}): " + ('<green>Proof search succeeded</green>' if is_success else '<yellow>failed</yellow>' + f' in {search_result.duration} (s)'))
            
            if search_result.success:
                log_debug(f'search({tag_i}): \n{str(init_solution_state)}\n{proof}')
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
                    sample['responses'] = [r.model_dump_json() for r in sample['responses']]
                    f.write(json.dumps(sample)+'\n')
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
