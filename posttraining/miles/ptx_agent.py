"""Harbor agent function with wall-clock trial bounds, for ``--custom-agent-function-path ptx_agent.run``.

Wraps ``swe_agent_function.run`` (examples/swe-agent-harbor-docker must be on PYTHONPATH) and stamps
``agent_metrics.trial_start_ts`` / ``trial_end_ts`` (``time.time()``) so ``ptx_rollout_log`` can compute
sandbox concurrency. Under the default ``--custom-agent-function-mode subproc`` this runs in a child
process that pickles ``run`` by reference, so this module must be importable from the rollout's sys.path.

``abort`` is re-exported: miles resolves the oversampling teardown hook as ``<agent module>.abort``
(``miles.utils.misc.call_agent_abort_hook``).
"""

import time
from typing import Any

from swe_agent_function import (
    abort,  # noqa: F401
    run as _swe_run,
)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    start = time.time()
    result = await _swe_run(
        base_url, prompt, request_kwargs=request_kwargs, metadata=metadata, **kwargs
    )
    end = time.time()
    if result is None:
        # swe_agent_function returns None when the agent-server call fails or times out; reward stays 0.0.
        result = {"exit_status": "AgentCallFailed", "agent_metrics": {}}
    result["agent_metrics"] = {
        **(result.get("agent_metrics") or {}),
        "trial_start_ts": start,
        "trial_end_ts": end,
    }
    return result
