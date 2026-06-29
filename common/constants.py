import regex as re

FPS_GLOBAL_SETTING = {
    'TO_SYNC_ENABLED' : False
}

REPL_TIMEOUT = 120
REPL_MAXREAD = 1000000
API_TIMEOUT = 3600

_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)

Expr = str


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
assert BANNED_TOKENS[0] == 'sorry'  # Keep `sorry` first for callers that rely on this ordering.

DEFAULT_CORE_OPTIONS = ["maxHeartbeats=0", "maxRecDepth=100000"]
CORE_OPTIONS = DEFAULT_CORE_OPTIONS + ["tactic.hygienic=false"] + ['pp.fullNames=true', 'pp.funBinderTypes=true', 'pp.piBinderTypes=true']
OPEN_HEADER = 'open BigOperators Real Nat Topology\n'   # This doesn't work in Pantograph tactic mode!
H_SUBMISSION_NAME = '_h_answer'
H_CONCLUSION_NAME = '_h_conclusion'

CODEBLOCK_PATTERN = re.compile(r'```(?:.*?)\n(.*?)```', flags=re.DOTALL)

RPE_TACTICS = ['norm_num', 'ring_nf', 'rfl', 'aesop'] # , 'rw_search']
