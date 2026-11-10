#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RAG 101 - corpus + answerability (script version of rag_101.ipynb).

The smallest end-to-end RAG demo: build a vector corpus, retrieve passages for
a query, and ask the model whether those passages can actually answer it.

    1. Build (first run) or load the ChromaDB corpus of government-service passages
    2. Optionally launch a local vLLM server for the Granite Switch model
    3. Connect mellea to the server and register the embedded adapters
    4. For each query: retrieve top-K passages -> rag.check_answerability

Usage:

    # Against a vLLM server that is already running (no GPU needed here)
    python tutorials/scripts/reference/rag_101.py --base-url http://localhost:8000/v1

    # Launch vLLM locally first (CUDA GPU + granite-switch[vllm] required)
    python tutorials/scripts/reference/rag_101.py --launch-server

    # Your own questions
    python tutorials/scripts/reference/rag_101.py -q "How do I renew a passport?" -q "..."

``--base-url`` and ``--model`` default to ``$VLLM_BASE_URL`` / ``$VLLM_MODEL_NAME``.
The corpus is cached in ``--chroma-path``, so only the first run embeds documents.

Requires: mellea, chromadb, sentence-transformers (all in ``granite-switch[tutorials]``),
plus vLLM for ``--launch-server``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rag_101")

DEFAULT_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"
DEFAULT_PORT = 8000

# The notebook's two demo queries: one in scope, one clearly out of scope.
DEFAULT_QUERIES = [
    "How do I check the status of my federal tax refund?",
    "What's the best recipe for sourdough bread?",
]

# Lines in the vLLM log that usually point at the real cause of a failed start.
_LOG_ERROR_PATTERN = re.compile(
    r"Error|Exception|out of memory|not enough|insufficient|not supported|not found",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# CLI and logging
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the rag_101 tutorial (corpus + answerability) as a script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    server = p.add_argument_group("vLLM server")
    server.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL"),
        help="OpenAI-compatible endpoint. Defaults to $VLLM_BASE_URL, "
        "else http://localhost:<port>/v1.",
    )
    server.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL),
        help="Model name as served by vLLM (GET /v1/models).",
    )
    server.add_argument(
        "--adapter-source",
        default=os.environ.get("GRANITE_SWITCH_SOURCE"),
        help="Hub repo id or local path with the adapters' I/O configs "
        "(defaults to --model).",
    )
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument(
        "--launch-server",
        action="store_true",
        help="Start vLLM for --model before running (needs a GPU); "
        "stopped again on exit.",
    )
    server.add_argument("--max-model-len", type=int, default=10240)
    server.add_argument(
        "--dtype",
        default=None,
        help="Passed to vLLM as --dtype (e.g. float16 on a T4, which lacks bfloat16).",
    )
    server.add_argument("--server-timeout", type=int, default=900, help="Seconds.")
    server.add_argument("--server-log", default="vllm_server.log")

    corpus = p.add_argument_group("corpus")
    corpus.add_argument("--chroma-path", default="./govt_chroma")
    corpus.add_argument("--jsonl-path", default="./govt.jsonl")
    corpus.add_argument(
        "--full-corpus",
        action="store_true",
        help="Embed all ~49k passages instead of the 177-passage tutorial subset "
        "(slow; delete --chroma-path first if it already holds the subset).",
    )
    corpus.add_argument(
        "--top-k", type=int, default=5, help="Passages retrieved per query."
    )

    run = p.add_argument_group("run")
    run.add_argument(
        "-q",
        "--query",
        action="append",
        dest="queries",
        help="Question to check (repeatable). Defaults to the notebook's two queries.",
    )
    run.add_argument(
        "--show-passages",
        type=int,
        default=3,
        help="How many retrieved passages to print per query.",
    )
    run.add_argument("--output", type=Path, help="Write all results as JSON here.")
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also show mellea / HTTP / library INFO logs.",
    )
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    """Timestamped logs for this script; libraries quiet unless --verbose."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        return
    # mellea reads this when it is first imported, so set it before any import.
    os.environ.setdefault("MELLEA_LOG_LEVEL", "WARNING")
    for name in ("httpx", "chromadb", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def section(title: str) -> None:
    log.info("")
    log.info("── %s %s", title, "─" * max(0, 66 - len(title)))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


# ---------------------------------------------------------------------------
# Step 1 - corpus
# ---------------------------------------------------------------------------


def load_corpus(args: argparse.Namespace):
    """Build (first run) or load the ChromaDB collection of govt passages."""
    from granite_switch.tutorials.chroma_loader import load_or_build_govt_chroma

    started = time.monotonic()
    collection = load_or_build_govt_chroma(
        chroma_path=args.chroma_path,
        jsonl_path=args.jsonl_path,
        load_only_tutorial_docs=not args.full_corpus,
    )
    log.info(
        "Corpus ready: %s passages in %s (%.1fs)",
        f"{collection.count():,}",
        args.chroma_path,
        time.monotonic() - started,
    )
    return collection


# ---------------------------------------------------------------------------
# Step 2 - vLLM server (only with --launch-server)
# ---------------------------------------------------------------------------


def show_server_errors(log_file: str, max_lines: int = 25) -> None:
    """Print the lines of the vLLM log that explain a failed start.

    The end of the log is only the generic "Engine core initialization failed";
    the cause (OOM, unsupported dtype, missing compiler, ...) is further up.
    """
    try:
        lines = Path(log_file).read_text(errors="replace").splitlines()
    except OSError as exc:
        log.error("Could not read %s: %s", log_file, exc)
        return

    hits = [line for line in lines if _LOG_ERROR_PATTERN.search(line)]
    log.error("Relevant lines from %s:", log_file)
    for line in hits[:max_lines]:
        log.error("  | %s", line.strip())
    log.error("Last lines of %s:", log_file)
    for line in lines[-10:]:
        log.error("  | %s", line.strip())


def start_server(args: argparse.Namespace) -> None:
    """Launch vLLM in the background and block until /health answers."""
    from granite_switch.tutorials.vllm_server import launch_vllm, wait_for_server

    extra_args = ["--dtype", args.dtype] if args.dtype else []
    proc = launch_vllm(
        model=args.model,
        port=args.port,
        log_file=args.server_log,
        max_model_len=args.max_model_len,
        extra_args=extra_args,
    )

    def stop() -> None:
        if proc.poll() is None:
            log.info("Stopping vLLM server (pid %d)", proc.pid)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    atexit.register(stop)

    started = time.monotonic()
    if not wait_for_server(
        args.port, timeout=args.server_timeout, log_file=args.server_log
    ):
        show_server_errors(args.server_log)
        sys.exit("vLLM server did not become healthy (see the log lines above).")
    log.info("vLLM server ready in %.0fs", time.monotonic() - started)


# ---------------------------------------------------------------------------
# Step 3 - mellea backend
# ---------------------------------------------------------------------------


def connect_backend(base_url: str, model: str, adapter_source: str):
    """mellea OpenAIBackend pointed at vLLM, with the embedded adapters registered."""
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model_id=model, base_url=base_url, api_key="unused")
    backend.register_embedded_adapter_model(adapter_source)
    adapters = backend.list_adapters()
    log.info("Registered %d adapters: %s", len(adapters), ", ".join(adapters))
    return backend


# ---------------------------------------------------------------------------
# Step 4 - retrieve + answerability
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    doc_id: str
    title: str
    url: str
    distance: float
    text: str


@dataclass
class QueryResult:
    query: str
    verdict: str
    seconds: float
    passages: list[Passage] = field(default_factory=list)


def retrieve(collection, query: str, top_k: int) -> list[Passage]:
    """Top-K passages from ChromaDB, most similar first."""
    hits = collection.query(query_texts=[query], n_results=top_k)
    return [
        Passage(
            doc_id=doc_id,
            title=(meta or {}).get("title", ""),
            url=(meta or {}).get("url", ""),
            distance=float(distance),
            text=" ".join(text.split()),
        )
        for doc_id, meta, distance, text in zip(
            hits["ids"][0],
            hits["metadatas"][0],
            hits["distances"][0],
            hits["documents"][0],
        )
    ]


def check_answerability(query: str, passages: list[Passage], backend) -> str:
    """'answerable' or 'unanswerable', from the answerability adapter."""
    from mellea.stdlib.components import Document
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    documents = [Document(doc_id=str(i), text=p.text) for i, p in enumerate(passages)]
    return rag.check_answerability(query, documents, ChatContext(), backend)


def run_query(query: str, collection, backend, top_k: int) -> QueryResult:
    started = time.monotonic()
    passages = retrieve(collection, query, top_k)
    verdict = check_answerability(query, passages, backend)
    return QueryResult(query, verdict, time.monotonic() - started, passages)


def report(result: QueryResult, show_passages: int) -> None:
    mark = "✅" if result.verdict == "answerable" else "🚫"
    log.info(
        "%s %-12s %s  (%.1fs)",
        mark,
        result.verdict.upper(),
        result.query,
        result.seconds,
    )
    for rank, p in enumerate(result.passages[:show_passages], start=1):
        preview = p.text if len(p.text) <= 140 else p.text[:137] + "..."
        log.info("     #%d  dist=%.3f  %s", rank, p.distance, p.title or p.doc_id)
        log.info("         %s", preview)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = parse_args(argv)
    setup_logging(args.verbose)

    base_url = args.base_url or f"http://localhost:{args.port}/v1"
    adapter_source = args.adapter_source or args.model
    queries = args.queries or DEFAULT_QUERIES

    log.info("vLLM endpoint : %s", base_url)
    log.info("Model         : %s", args.model)
    log.info(
        "Corpus        : %s (%s)",
        args.chroma_path,
        "full" if args.full_corpus else "tutorial subset",
    )

    # Corpus first: on a first run its embedding briefly uses the GPU, and the
    # loader frees that memory before vLLM claims it.
    section("1 · Corpus")
    collection = load_corpus(args)

    if args.launch_server:
        section("2 · Launch vLLM server")
        start_server(args)

    section("3 · Connect mellea")
    backend = connect_backend(base_url, args.model, adapter_source)

    section("4 · Retrieve + answerability")
    results = []
    for query in queries:
        result = run_query(query, collection, backend, args.top_k)
        report(result, args.show_passages)
        results.append(result)

    section("Summary")
    answerable = sum(r.verdict == "answerable" for r in results)
    log.info("%d / %d queries answerable from the corpus", answerable, len(results))

    if args.output:
        args.output.write_text(json.dumps([asdict(r) for r in results], indent=2))
        log.info("Results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
