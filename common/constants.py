from typing import List

import regex as re

FPS_GLOBAL_SETTING = {
    'TO_SYNC_ENABLED' : False
}

INIT_WAIT_TIME = 5
REPL_TIMEOUT = 120
REPL_MAXREAD = 1000000
RETRY_WAIT_TIME = 10
API_TIMEOUT = 3600

LOCK_WAIT_TIMEOUT = 600
LOCK_TRY_DELAY = 0.05

_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)
TERMINAL_TEXT = ''

Expr = str

PROBLEM_KEYS = set(['informal_problem', 'informal_answer', 'informal_solution'])

# For generator
BANNED_TOKENS = [
        'sorry',
        'admit',
        'stop',
        
        'slim_check',   # If slim_check successfully tests 100 examples, it acts like admit. If it gives up or finds a counter-example, it reports an error.
        
        # Tactics that may break the proof state by forcefully closing the goals
        'apply?',       # `rw?`` should not be left in proofs; it is a search tool, like `apply?`.
        'rw?',
        'hint',
        'rw_search',
        
        # Tactics that are also search tools, but does not break proof states
        'change?',
        'have?',
        'says',
        'try_this',
        'unfold?',
        
        # unsafe axioms: even though they are not sound, the kernel will not let us use them for regular proofs.
        'lcErased',
        'lcProof',
        'lcCast',
        'lcUnreachable',
        'Quot.lcInv'
    ]
assert BANNED_TOKENS[0] == 'sorry'  # The first banned token should be `sorry` (required by `agent/solution_search.py`)

DEFAULT_CORE_OPTIONS = ["maxHeartbeats=0", "maxRecDepth=100000"]
CORE_OPTIONS = DEFAULT_CORE_OPTIONS + ["tactic.hygienic=false"] + ['pp.fullNames=true', 'pp.funBinderTypes=true', 'pp.piBinderTypes=true']
PARSING_EXTRA_OPTIONS = ['pp.numericTypes=true', 'pp.structureInstances=false', 'pp.safeShadowing=false', 'pp.fieldNotation.generalized=false', 'pp.explicit=true', 'pp.deepTerms=true', 'pp.proofs=true', f'pp.maxSteps={REPL_MAXREAD}', 'pp.notation=false']
OPEN_HEADER = 'open BigOperators Real Nat Topology\n'   # This doesn't work in Pantograph tactic mode!
H_SUBMISSION_NAME = 'h_submission'

SOLUTION_PRUNING_HEADER = '''set_option linter.unusedTactic true
set_option linter.unusedVariables true
set_option linter.unusedVariables.analyzeTactics true
'''

NEW_LINE = '\n'
ANSWER_PATTERN = re.compile(r"\\boxed{(.*?)}")
CODEBLOCK_PATTERN = re.compile(r'```(?:.*?)\n(.*?)```', flags=re.DOTALL)

RPE_TACTICS = ['rfl', 'norm_num', 'ring_nf', 'rw_search', 'aesop']
