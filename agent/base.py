from dataclasses import dataclass, field
from typing import Any, Optional

from common.pantograph.dataclasses import FormalProblem, GoalState


@dataclass
class ProofAttempt:
    success: bool
    duration: float
    proof: Any = None
    response_content: Optional[str] = None
    error: Optional[str] = None
    final_state: Optional[GoalState] = None
    metadata: dict = field(default_factory=dict)

    def serialize(self) -> dict:
        return {
            "success": self.success,
            "duration": self.duration,
            "proof": self.proof,
            "response_content": self.response_content,
            "error": self.error,
            "final_state": None if self.final_state is None else self.final_state.serialize(),
            "metadata": self.metadata,
        }


@dataclass
class ProofResult:
    success: bool
    duration: float
    proof: Any = None
    final_state: Optional[GoalState] = None
    attempts: list[ProofAttempt] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def serialize(self) -> dict:
        return {
            "success": self.success,
            "duration": self.duration,
            "proof": self.proof,
            "final_state": None if self.final_state is None else self.final_state.serialize(),
            "attempts": [attempt.serialize() for attempt in self.attempts],
            "metadata": self.metadata,
        }


class ProverAgent:
    agent_name = "prover"

    async def prove_async(
        self,
        sample: FormalProblem,
        server,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
    ) -> ProofResult:
        raise NotImplementedError



