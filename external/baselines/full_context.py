from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from external.baselines.common import (
    JsonlWriter,
    add_common_args,
    already_done,
    base_record,
    call_answer_model,
    format_full_context,
    load_filtered_examples,
    make_clients,
    now,
    simple_answer_messages,
    write_record,
)


METHOD = "external_full_context_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external full-context baseline.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clients = make_clients(args.config, include_embedding=False)
    examples = load_filtered_examples(args)
    done = already_done(args.out, overwrite=args.overwrite)
    writer = JsonlWriter(args.out)

    pending = [(index, example) for index, example in enumerate(examples, start=1) if example.sample_id not in done]
    for index, example in enumerate(examples, start=1):
        if example.sample_id in done:
            print(f"[{index}/{len(examples)}] skip {example.sample_id}", flush=True)

    def run_one(index: int, example: object) -> None:
        started = now()
        try:
            context = format_full_context(example.turns, max_chars=args.max_context_chars)
            hypothesis, raw_response, tokens = call_answer_model(
                clients.answer,
                clients.config,
                simple_answer_messages(example, context),
            )
            record = base_record(
                example,
                method=METHOD,
                config=clients.config,
                query_tokens=tokens,
                query_time_seconds=now() - started,
                hypothesis=hypothesis,
                raw_response=raw_response,
                extra={
                    "context_mode": "full_context",
                    "context_chars": len(context),
                    "num_turns": len(example.turns),
                },
            )
        except Exception as exc:
            record = base_record(
                example,
                method=METHOD,
                config=clients.config,
                query_time_seconds=now() - started,
                error=repr(exc),
            )
            if args.stop_on_error:
                raise

        writer.write(record)
        print(f"[{index}/{len(examples)}] done {example.sample_id}", flush=True)

    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for index, example in pending:
            run_one(index, example)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, index, example) for index, example in pending]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
