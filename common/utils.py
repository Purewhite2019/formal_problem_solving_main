import os
import os.path as osp
import time
import errno
from pathlib import Path
import regex as re
from typing import Union, Tuple, List, Dict, Any, Optional, Generator, AsyncGenerator, Set, Iterable
from contextlib import contextmanager, asynccontextmanager
import asyncio
import collections as C
import itertools as I
import functools as F
import inspect
import signal

import ipdb
import aiofiles
from loguru import logger
import pexpect

from common.constants import Expr, _SPACES_REGEX, CODEBLOCK_PATTERN, FPS_GLOBAL_SETTING


replace_calc = lambda s: re.sub(r'by\s+calc', r'calc', s)
replace_sorry = lambda s: re.sub(r'by\s+sorry', r'sorry', s)

def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)

def format_variable_sequence(s : Iterable['Variable']) -> str:
    return ' '.join([f'({v.name} : {v.t})' if v.name not in [None, '_'] else f'[{v.t}]' for v in s])

def inplace_add(s: str, ss: List[str]) -> str:
    ss.append(s)
    return s

def extract_code(s: str) -> str:
    parse_result = re.findall(CODEBLOCK_PATTERN, s)
    if len(parse_result) > 0:
        step = parse_result[0].strip()
    else:
        split_cnt = len(re.findall('```', s))
        if split_cnt == 0:
            step = s.strip()
        if split_cnt == 1:
            step = s[:s.find('```')].strip()
        else:
            step = s

    return step

def to_sync(func):
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        if not FPS_GLOBAL_SETTING['TO_SYNC_ENABLED']:
            raise RuntimeError('to_sync() is not enabled in common.constants.FPS_GLOBAL_SETTING')
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    return wrapper

class Spawn(pexpect.spawn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delayafterclose = 0.5
        self.delayafterterminate = 0.5
        self.ptyproc.delayafterclose = 0.5
        self.ptyproc.delayafterterminate = 0.5
        # self.delaybeforesend = None

    async def send_async(self, s):
        if self.delaybeforesend is not None:
            await asyncio.sleep(self.delaybeforesend)

        s = self._coerce_send_string(s)
        self._log(s, 'send')

        b = self._encoder.encode(s, final=False)

        return os.write(self.child_fd, b)        
        # while b:
        #     try:
        #         bytes_written = os.write(self.child_fd, b)
        #         b = b[bytes_written:]
        #     except BlockingIOError:
        #         await asyncio.sleep(0)
        #         pass
        
        # return bytes_written  

    async def sendline_async(self, s=''):
        '''Wraps send(), sending string ``s`` to child process, with
        ``os.linesep`` automatically appended. Returns number of bytes
        written.  Only a limited number of bytes may be sent for each
        line in the default terminal mode, see docstring of :meth:`send`.
        '''
        s = self._coerce_send_string(s)
        return await self.send_async(s + self.linesep)

    async def read_async(self, size=-1):
        '''This reads at most "size" bytes from the file (less if the read hits
        EOF before obtaining size bytes). If the size argument is negative or
        omitted, read all data until EOF is reached. The bytes are returned as
        a string object. An empty string is returned when EOF is encountered
        immediately. '''

        if size == 0:
            return self.string_type()
        if size < 0:
            # delimiter default is EOF
            await self.expect(self.delimiter, async_=True)
            return self.before

        # I could have done this more directly by not using expect(), but
        # I deliberately decided to couple read() to expect() so that
        # I would catch any bugs early and ensure consistent behavior.
        # It's a little less efficient, but there is less for me to
        # worry about if I have to later modify read() or expect().
        # Note, it's OK if size==-1 in the regex. That just means it
        # will never match anything in which case we stop only on EOF.
        cre = re.compile(self._coerce_expect_string('.{%d}' % size), re.DOTALL)
        # delimiter default is EOF
        index = await self.expect([cre, self.delimiter], async_=True)
        if index == 0:
            ### FIXME self.before should be ''. Should I assert this?
            return self.after
        return self.before

    async def readline_async(self, size=-1):
        '''This reads and returns one entire line. The newline at the end of
        line is returned as part of the string, unless the file ends without a
        newline. An empty string is returned if EOF is encountered immediately.
        This looks for a newline as a CR/LF pair (\\r\\n) even on UNIX because
        this is what the pseudotty device returns. So contrary to what you may
        expect you will receive newlines as \\r\\n.

        If the size argument is 0 then an empty string is returned. In all
        other cases the size argument is ignored, which is not standard
        behavior for a file-like object. '''

        if size == 0:
            return self.string_type()
        # delimiter default is EOF
        index = await self.expect([self.crlf, self.delimiter], async_=True)
        if index == 0:
            return self.before + self.crlf
        else:
            return self.before

    async def terminate_async(self, force=False):
        '''This forces a child process to terminate. It starts nicely with
        SIGHUP and SIGINT. If "force" is True then moves onto SIGKILL. This
        returns True if the child was terminated. This returns False if the
        child could not be terminated. '''

        if not self.isalive():
            return True
        try:
            self.kill(signal.SIGHUP)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGCONT)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGINT)
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            if force:
                self.kill(signal.SIGKILL)
                await asyncio.sleep(self.delayafterterminate)
                if not self.isalive():
                    return True
                else:
                    return False
            return False
        except OSError:
            # I think there are kernel timing issues that sometimes cause
            # this to happen. I think isalive() reports True, but the
            # process is dead to the kernel.
            # Make one last attempt to see if the kernel is up to date.
            await asyncio.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            else:
                return False


def unique(r : List) -> List:
    s = []
    for i in r:
        if i not in s:
            s.append(i)
    return s

def parse_expr(payload: Dict) -> Expr:
    """
    :meta private:
    """
    return payload["pp"]

def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()

def remove_comments(code):
    code = re.sub(r'/-(.|\n)*?-/', '', code)
    code = re.sub(r'--.*', '', code)
    return code

def pdb_on_assertion_error(func):
    @F.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AssertionError as e:
            ipdb.set_trace()
            raise e
    return wrapper

def remove_min_whitespace(s: str) -> str:
    lines = s.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return s
    min_whitespace = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    result = []
    for line in lines:
        if line.strip():
            result.append(line[min_whitespace:])
        else:
            result.append(line)
    return '\n'.join(result)
