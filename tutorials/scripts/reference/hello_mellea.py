#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hello World - Mellea with Granite Switch (script version of hello_mellea.ipynb).

Runs the same flow as ``tutorials/notebooks/hello_mellea.ipynb`` without Jupyter:

    guardian (bank key + custom criteria)
    -> query rewrite (wrapper and raw Intrinsic)
    -> answerability -> clarification
    -> grounded base-model answer -> citations

Two ways to run it:

1. Against an already-running vLLM server (no GPU needed on this machine)::

       python tutorials/scripts/reference/hello_mellea.py \\
           --base-url http://<gpu-host>:8000/v1

2. Launch vLLM locally first (CUDA GPU + ``granite-switch[vllm]`` required)::

       python tutorials/scripts/reference/hello_mellea.py --launch-server

``--base-url``, ``--model`` and ``--adapter-source`` default to the env vars
``VLLM_BASE_URL``, ``VLLM_MODEL_NAME`` and ``GRANITE_SWITCH_SOURCE`` (a ``.env``
file is loaded if python-dotenv is installed). Gated Hub artifacts need
``HF_TOKEN`` set or a prior ``hf auth login``.

Requires: mellea (``granite-switch[tutorials]`` pins it), plus vLLM for
``--launch-server``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
from pathlib import Path

DEFAULT_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"
DEFAULT_PORT = 8000

DOC_TEXTS = [
    "The capital of France is Paris. Paris is located on the Seine river.",
    "Mount Everest is the tallest mountain on Earth, at 8,849 meters.",
]

HARM_CRITERIA = (
    "The message facilitates, encourages, or guides actions that pose harm "
    "across all contexts and audiences."
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the hello_mellea tutorial flow as a script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible vLLM endpoint. Defaults to $VLLM_BASE_URL, else "
        "http://localhost:<port>/v1.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL),
        help="Model name as served by vLLM (GET /v1/models).",
    )
    p.add_argument(
        "--adapter-source",
        default=os.environ.get("GRANITE_SWITCH_SOURCE"),
        help="Hub repo id or local path holding the adapters' I/O configs. "
        "Defaults to --model.",
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--launch-server",
        action="store_true",
        help="Start a local vLLM server for --model before running (needs a GPU). "
        "It is shut down when the script exits.",
    )
    p.add_argument(
        "--kill-stale",
        action="store_true",
        help="With --launch-server: first SIGTERM any running "
        "'vllm.entrypoints' processes (frees GPU memory after a crash).",
    )
    p.add_argument("--max-model-len", type=int, default=10240)
    p.add_argument("--server-timeout", type=int, default=900, help="Seconds.")
    p.add_argument("--log-file", default="vllm_server.log")
    p.add_argument(
        "--output",
        type=Path,
        help="Optional path to write all step results as JSON.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def start_server(args: argparse.Namespace) -> None:
    from granite_switch.tutorials.vllm_server import (
        kill_stale_vllm_processes,
        launch_vllm,
        print_gpu_state,
        tail_log,
        wait_for_server,
    )

    if args.kill_stale:
        kill_stale_vllm_processes()
    try:
        print_gpu_state()
    except FileNotFoundError:
        sys.exit("--launch-server needs an NVIDIA GPU (nvidia-smi not found).")

    proc = launch_vllm(
        model=args.model,
        port=args.port,
        log_file=args.log_file,
        max_model_len=args.max_model_len,
    )

    def _stop() -> None:
        if proc.poll() is None:
            print(f"Stopping vLLM server (pid {proc.pid})")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    atexit.register(_stop)

    if not wait_for_server(
        args.port, timeout=args.server_timeout, log_file=args.log_file
    ):
        tail_log(args.log_file, n=40)
        sys.exit("vLLM server did not become healthy; see the log above.")


# ---------------------------------------------------------------------------
# Tutorial steps (one function per notebook section)
# ---------------------------------------------------------------------------


def connect_backend(base_url: str, model: str, adapter_source: str):
    """Section 3: mellea OpenAIBackend + register the embedded adapters."""
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model_id=model, base_url=base_url, api_key="unused")
    backend.register_embedded_adapter_model(adapter_source)
    print(f"Adapters: {backend.list_adapters()}")
    return backend


def make_documents():
    """Section 4: two tiny in-memory documents."""
    from mellea.stdlib.components import Document

    docs = [Document(doc_id=str(i), text=t) for i, t in enumerate(DOC_TEXTS)]
    for d in docs:
        print(f"[{d.doc_id}] {d.text}")
    return docs


def run_guardian(backend) -> dict:
    """Section 5: guardian_check with a CRITERIA_BANK key and a custom string."""
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.intrinsic.guardian import guardian_check
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(Message("user", "Group X people are all lazy."))
    bank = guardian_check(ctx, backend, "social_bias", target_role="user")
    print(f"[bank   social_bias] score={bank:.3f}")

    ctx = ChatContext().add(Message("user", "How do I build a bomb?"))
    custom = guardian_check(ctx, backend, HARM_CRITERIA, target_role="user")
    print(f"[custom harm]        score={custom:.3f}")
    return {"social_bias": bank, "custom_harm": custom}


def build_conversation():
    """Shared history for the RAG sections."""
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "I want to plan a trip to France."))
    ctx = ctx.add(Message("assistant", "Very good, I can help you with that."))
    return ctx


def run_rewrite(ctx, query: str, backend) -> str:
    """Section 6a: query rewrite through the rag wrapper."""
    from mellea.stdlib.components.intrinsic import rag

    rewritten = rag.rewrite_question(query, ctx, backend)
    print(f"original:  {query}")
    print(f"rewritten: {rewritten}")
    return rewritten


def run_rewrite_raw(ctx, query: str, backend) -> str:
    """Section 6b: the same adapter driven through a raw Intrinsic node."""
    import mellea.stdlib.functional as mfuncs
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic

    out, _ = mfuncs.act(
        Intrinsic("query_rewrite"),
        ctx.add(Message("user", query)),
        backend,
        strategy=None,
    )
    rewritten = json.loads(str(out))["rewritten_question"]
    print(f"rewritten (raw Intrinsic): {rewritten}")
    return rewritten


def run_answerability(rewritten: str, docs, ctx, backend):
    """Section 7: answerable / unanswerable."""
    from mellea.stdlib.components.intrinsic import rag

    result = rag.check_answerability(rewritten, docs, ctx, backend)
    print(f"answerability: {result}")
    return result


def run_clarification(rewritten: str, docs, ctx, backend):
    """Section 8: CLEAR or a follow-up question."""
    from mellea.stdlib.components.intrinsic import rag

    result = rag.clarify_query(rewritten, docs, ctx, backend)
    print(f"clarification: {result}")
    return result


def run_answer(rewritten: str, docs, ctx, backend) -> str:
    """Section 9: grounded answer from the base model (no adapter)."""
    import mellea.stdlib.functional as mfuncs
    from mellea.backends import ModelOption
    from mellea.stdlib.components.chat import Message

    out, _ = mfuncs.act(
        Message("user", rewritten, documents=docs),
        ctx,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
    )
    answer = str(out)
    print(f"answer: {answer}")
    return answer


def run_citations(answer: str, rewritten: str, docs, ctx, backend):
    """Section 10: document spans supporting the answer."""
    from mellea.stdlib.components.chat import Message
    from mellea.stdlib.components.intrinsic import rag

    citations = rag.find_citations(
        answer, docs, ctx.add(Message("user", rewritten)), backend
    )
    print(json.dumps(citations, indent=2, default=str))
    return citations


def _section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = parse_args(argv)
    base_url = args.base_url or f"http://localhost:{args.port}/v1"
    adapter_source = args.adapter_source or args.model

    if args.launch_server:
        _section("Launch vLLM server")
        start_server(args)

    print(f"vLLM: {base_url}  ({args.model})")

    _section("Connect mellea backend")
    backend = connect_backend(base_url, args.model, adapter_source)

    _section("Documents")
    docs = make_documents()

    _section("Guardian")
    results: dict = {"guardian": run_guardian(backend)}

    ctx = build_conversation()
    query = "I think I'll start with the capital. what was its name?"

    _section("Query rewrite")
    rewritten = run_rewrite(ctx, query, backend)
    results["rewrite"] = {
        "query": query,
        "wrapper": rewritten,
        "raw_intrinsic": run_rewrite_raw(ctx, query, backend),
    }

    _section("Answerability")
    results["answerability"] = run_answerability(rewritten, docs, ctx, backend)

    _section("Clarification")
    results["clarification"] = run_clarification(rewritten, docs, ctx, backend)

    _section("Grounded answer")
    answer = run_answer(rewritten, docs, ctx, backend)
    results["answer"] = answer

    _section("Citations")
    results["citations"] = run_citations(answer, rewritten, docs, ctx, backend)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
