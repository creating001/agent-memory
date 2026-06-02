from __future__ import annotations

import argparse
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from agent_memory.baseline.pipeline import METHOD_NAME, NaiveRagBaseline
from agent_memory.baseline.store import store_exists
from agent_memory.core.config import get_config, load_config
from agent_memory.core.embedding import EmbeddingClient
from agent_memory.core.io import TeeLogger, append_jsonl, completed_ids, load_env_file
from agent_memory.core.llm import ChatClient
from agent_memory.core.schema import Example
from agent_memory.datasets import load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run the Agent Memory LTS baseline: {METHOD_NAME}.")
    parser.add_argument("--config", type=Path, default=Path("src/agent_memory/configs/base.yaml"))
    parser.add_argument("--dataset", choices=("auto", "longmemeval", "locomo"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s_cleaned.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("full", "build", "query"), default="full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel memory groups. Defaults: LoCoMo=2, LongMemEval=4.",
    )
    return parser.parse_args()


def make_baseline(config: dict[str, Any], answer_api_key: str) -> NaiveRagBaseline:
    embedding_cfg = config["embedding"]
    answer_cfg = config["answer"]
    return NaiveRagBaseline(
        embedding_client=EmbeddingClient(
            model=str(embedding_cfg["name"]),
            base_url=str(embedding_cfg["base_url"]),
            api_key=str(embedding_cfg.get("api_key", "EMPTY")),
            batch_size=int(embedding_cfg.get("batch_size", 64)),
            normalize=bool(embedding_cfg.get("normalize", True)),
            query_instruction=str(embedding_cfg.get("query_instruction", "")),
            max_input_bytes=int(embedding_cfg.get("max_input_bytes", 0)),
        ),
        answer_client=ChatClient(
            api_key=answer_api_key,
            base_url=str(get_config(config, "answer.base_url", "")),
            timeout_seconds=float(answer_cfg.get("timeout_seconds", 120)),
            max_retries=int(answer_cfg.get("max_retries", 2)),
            default_seed=optional_int(get_config(config, "answer.seed")),
            default_top_p=optional_float(get_config(config, "answer.top_p")),
        ),
        config=config,
    )


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def group_by_memory(examples: list[Example]) -> list[tuple[str, list[tuple[int, Example]]]]:
    groups: OrderedDict[str, list[tuple[int, Example]]] = OrderedDict()
    for index, example in enumerate(examples, start=1):
        groups.setdefault(example.memory_id, []).append((index, example))
    return list(groups.items())


def default_workers(examples: list[Example]) -> int:
    if examples and examples[0].dataset == "locomo":
        return 2
    return 4


def error_record(example: Example, exc: Exception) -> dict[str, Any]:
    return {
        "sample_id": example.sample_id,
        "memory_id": example.memory_id,
        "dataset": example.dataset,
        "question": example.question,
        "answer": example.answer,
        "hypothesis": "",
        "method": METHOD_NAME,
        "error": repr(exc),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    load_env_file(str(get_config(config, "paths.env_file", ".env")))

    out = args.out or Path(str(get_config(config, "paths.prediction_file")))
    store_root = args.store_root or Path(str(get_config(config, "paths.store_root")))
    answer_api_key = os.environ.get(str(get_config(config, "answer.api_key_env", "LOCAL_LLM_API_KEY")), "")
    if not answer_api_key:
        raise ValueError("Missing answer model API key environment variable.")

    examples = load_examples(args.data, dataset=args.dataset)
    if args.start:
        examples = examples[args.start :]
    if args.limit:
        examples = examples[: args.limit]

    if args.overwrite and out.exists() and args.mode != "build":
        out.unlink()
    if args.overwrite and args.log_file and args.log_file.exists():
        args.log_file.unlink()
    done = set() if args.overwrite or args.mode == "build" else completed_ids(out)
    groups = group_by_memory(examples)
    workers = max(1, args.workers or default_workers(examples))

    with TeeLogger(args.log_file) as logger:
        log_lock = Lock()
        write_lock = Lock()
        total = len(examples)

        def log(message: str) -> None:
            with log_lock:
                logger.log(message)

        def write_record(record: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(out, record)

        def run_group(memory_id: str, group: list[tuple[int, Example]]) -> None:
            pending = [
                (index, example)
                for index, example in group
                if args.mode == "build" or example.sample_id not in done
            ]
            if args.mode != "build":
                for index, example in group:
                    if example.sample_id in done:
                        log(f"[{index}/{total}] skip {example.sample_id}")
                if not pending:
                    return

            baseline = make_baseline(config, answer_api_key)
            store_dir = store_root / memory_id

            try:
                if args.mode in {"full", "build"}:
                    if args.overwrite and store_dir.exists():
                        shutil.rmtree(store_dir)
                    if args.overwrite or not store_exists(store_dir):
                        stats = baseline.build_memory(group[0][1], store_dir)
                        log(f"[memory {memory_id}] built chunks={stats['num_chunks']}")
            except Exception as exc:
                if args.mode != "build":
                    for _, example in pending:
                        write_record(error_record(example, exc))
                log(f"[memory {memory_id}] build error: {exc}")
                if args.stop_on_error:
                    raise
                return

            if args.mode == "build":
                return

            for index, example in pending:
                try:
                    record = baseline.answer(example, store_dir)
                    write_record(record)
                    log(f"[{index}/{total}] done {example.sample_id} answer={record['hypothesis'][:80]!r}")
                except Exception as exc:
                    write_record(error_record(example, exc))
                    log(f"[{index}/{total}] error {example.sample_id}: {exc}")
                    if args.stop_on_error:
                        raise

        log(
            f"run start method={METHOD_NAME} dataset={args.dataset} data={args.data} mode={args.mode} "
            f"records={len(examples)} memories={len(groups)} workers={workers} "
            f"out={out} store_root={store_root}"
        )

        if workers == 1 or len(groups) <= 1:
            for memory_id, group in groups:
                run_group(memory_id, group)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_group, memory_id, group): memory_id for memory_id, group in groups}
                for future in as_completed(futures):
                    memory_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log(f"[memory {memory_id}] failed: {exc}")
                        if args.stop_on_error:
                            raise

        if args.mode == "build":
            log(f"build done: {len(groups)} memory stores")
        log("run done")


if __name__ == "__main__":
    main()
