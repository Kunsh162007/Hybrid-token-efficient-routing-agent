"""Command-line interface: route, run, eval, train-router, serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from routing_agent.config import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        return args.handler(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routing-agent",
        description="Hybrid token-efficient routing agent (local Gemma + Fireworks AI)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(required=True)

    p_route = sub.add_parser("route", help="Route a single task")
    p_route.add_argument("prompt")
    p_route.add_argument("--json", action="store_true", help="Emit JSON result")
    p_route.set_defaults(handler=_cmd_route)

    p_run = sub.add_parser("run", help="Route every task in a JSONL file")
    p_run.add_argument("--tasks", required=True)
    p_run.set_defaults(handler=_cmd_run)

    p_eval = sub.add_parser("eval", help="Evaluate accuracy vs. token spend")
    p_eval.add_argument("--tasks", required=True)
    p_eval.add_argument("--train-log", default=None, help="Append router training records")
    p_eval.set_defaults(handler=_cmd_eval)

    p_train = sub.add_parser("train-router", help="Train the learned router")
    p_train.add_argument("--log", required=True, help="Training records JSONL")
    p_train.add_argument("--out", required=True, help="Output model path")
    p_train.set_defaults(handler=_cmd_train)

    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(handler=_cmd_serve)

    p_submit = sub.add_parser(
        "submit", help="Hackathon harness mode: tasks.json in, results.json out"
    )
    p_submit.add_argument("--input", default=None, help="Path to tasks.json")
    p_submit.add_argument("--output", default=None, help="Path to results.json")
    p_submit.add_argument(
        "--time-budget", type=float, default=None,
        help="Global wall-clock budget in seconds",
    )
    p_submit.set_defaults(handler=_cmd_submit)

    return parser


def _result_payload(result) -> dict:
    return {
        "answer": result.answer,
        "exit_rung": result.exit_rung.name,
        "confidence": round(result.confidence, 3),
        "remote_tokens": result.remote_tokens,
        "task_type": str(result.task_type),
        "cached": result.cached,
        "verified": result.verified,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "trace": [
            {"rung": t.rung.name, "action": t.action, "detail": t.detail,
             "remote_tokens": t.remote_tokens}
            for t in result.trace
        ],
    }


def _cmd_route(args) -> int:
    from routing_agent.runtime import build_runtime

    runtime = build_runtime(args.config)
    result = runtime.route_task(args.prompt)
    if args.json:
        print(json.dumps(_result_payload(result), indent=2))
    else:
        print(result.answer)
        print(
            f"\n[exit={result.exit_rung.name} remote_tokens={result.remote_tokens} "
            f"confidence={result.confidence:.2f} verified={result.verified}]",
            file=sys.stderr,
        )
    return 0


def _cmd_run(args) -> int:
    from routing_agent.eval.harness import load_tasks
    from routing_agent.runtime import build_runtime

    runtime = build_runtime(args.config)
    for task in load_tasks(args.tasks):
        result = runtime.route_task(task.prompt)
        print(json.dumps({"id": task.id, **_result_payload(result)}))
    stats = runtime.budget.snapshot()
    print(
        f"[run] tasks={stats.tasks_completed} remote_tokens={stats.remote_tokens_spent} "
        f"free={stats.free_task_ratio:.1%}",
        file=sys.stderr,
    )
    return 0


def _cmd_eval(args) -> int:
    from routing_agent.eval.harness import load_tasks, run_eval
    from routing_agent.runtime import build_runtime

    runtime = build_runtime(args.config)
    report = run_eval(
        runtime.ladder, load_tasks(args.tasks), training_log_path=args.train_log
    )
    print(report.summary())
    failures = [row for row in report.rows if not row.correct]
    if failures:
        print("failed tasks:")
        for row in failures:
            answer_preview = " ".join(row.answer.split())[:70]
            print(f"  {row.task_id} [{row.exit_rung.name}]: {answer_preview!r}")
    return 0


def _cmd_train(args) -> int:
    from routing_agent.router.learned import LearnedRouter, LearnedRouterUnavailable

    try:
        router = LearnedRouter.train_from_log(args.log)
    except LearnedRouterUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    router.save(args.out)
    print(f"Trained router saved to {args.out}")
    print("Enable it via learned_router.enabled: true in config.yaml")
    return 0


def _cmd_submit(args) -> int:
    from routing_agent import submission

    kwargs = {}
    if args.input is not None:
        kwargs["input_path"] = args.input
    if args.output is not None:
        kwargs["output_path"] = args.output
    if args.time_budget is not None:
        kwargs["time_budget_seconds"] = args.time_budget
    return submission.run_submission(config_path=args.config, **kwargs)


def _cmd_serve(args) -> int:
    import uvicorn

    from routing_agent.config import load_config

    config = load_config(args.config)
    uvicorn.run(
        "routing_agent.web.app:create_default_app",
        factory=True,
        host=args.host or config.web.host,
        port=args.port or config.web.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
