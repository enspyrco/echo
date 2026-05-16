"""LangChain ``BaseChatModel`` backed by the Claude Code CLI (``claude --print``).

Why this exists: we want LangChain's routing primitives (RunnableBranch,
RunnableParallel, .with_fallbacks(), LCEL pipes) but the experiment is
billed against a Claude Max subscription, not the Anthropic API. The
Claude Code CLI authenticates against Max, so shelling out to
``claude --print --model <alias> "<prompt>"`` gives us inference at zero
marginal API spend — at the cost of ~3s of CLI startup overhead per call.

Tradeoffs vs ChatAnthropic:
  - No streaming (claude --print returns the full response on stdout)
  - No token-usage metadata returned by default (we'd have to parse it)
  - Each call spawns a subprocess; not suitable for tight inner loops
  - In exchange: zero API key management, zero per-token billing
"""

from __future__ import annotations

import subprocess
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ChatClaudeCode(BaseChatModel):
    """Chat model that routes inference through the Claude Code CLI."""

    model: str = Field(default="sonnet", description="Model alias: haiku | sonnet | opus")
    timeout: int = Field(default=180, description="Subprocess timeout in seconds")
    binary: str = Field(default="claude", description="Path/name of the claude binary")

    @property
    def _llm_type(self) -> str:
        return "chat-claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def _format(self, messages: List[BaseMessage]) -> str:
        """Flatten LangChain messages into a single prompt string.

        ``claude --print`` takes one prompt argument; it doesn't expose
        the multi-turn / role-tagged interface that the Anthropic API
        offers. We render messages as labelled sections so the model
        still sees the structure, even if it's not formally a multi-turn
        exchange. For most routing experiments this is fine — a single
        system+user turn is typical.
        """
        parts: list[str] = []
        for m in messages:
            role = (m.type or "user").upper()
            parts.append(f"[{role}]\n{m.content}")
        return "\n\n".join(parts)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._format(messages)
        try:
            result = subprocess.run(
                [self.binary, "--print", "--model", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude --print timed out after {self.timeout}s (model={self.model})"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"claude --print failed (rc={result.returncode}, model={self.model}): "
                f"{result.stderr.strip() or '<no stderr>'}"
            )

        text = result.stdout.rstrip()
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))],
        )
