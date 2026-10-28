#!/usr/bin/env python
"""CLI entry point.

  python main.py list-stages
  python main.py run --data data/emails.csv                       # full pipeline
  python main.py run --data data/emails.csv --stages baselines    # selected stage(s) (+ dependencies)
  python main.py run --data data/emails.csv --skip framework faithfulness attack demo   # no LLM needed
  python main.py run --data data/emails.csv --set epochs=3 n_eval_per_class=40 --reuse-models
  python main.py analyze --text "Dear customer, verify your account at http://paypa1-login.xyz"
"""
from __future__ import annotations

import argparse
import sys

from phishing_detection.config import Config
from phishing_detection.pipeline import ORDER, BY_NAME, Pipeline, analyze_text


def build_cfg(args) -> Config:
    cfg = Config.from_json(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "data", None):
        cfg.data_path = args.data
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    if getattr(args, "reuse_models", False):
        cfg.reuse_models = True
    if getattr(args, "no_save_models", False):
        cfg.save_models = False
    return cfg.override(getattr(args, "set", None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Explainable & robust phishing detection pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-stages", help="show pipeline stages and dependencies")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="JSON config file (see configs/default.json)")
    common.add_argument("--output-dir", help="where tables/figures/models/logs are written (default: outputs)")
    common.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values, e.g. epochs=3")

    r = sub.add_parser("run", parents=[common], help="run the pipeline")
    r.add_argument("--data", help="CSV/XLSX with 'Email Text' and 'Email Type' columns")
    r.add_argument("--stages", nargs="*", default=["all"], help=f"stages to run (default: all). Choices: {ORDER}")
    r.add_argument("--skip", nargs="*", default=[], help="stages to leave out")
    r.add_argument("--reuse-models", action="store_true", help="load saved DistilBERT models instead of retraining")
    r.add_argument("--no-save-models", action="store_true", help="do not write model weights to disk")
    r.add_argument("--continue-on-error", action="store_true", help="keep going when a stage fails")

    a = sub.add_parser("analyze", parents=[common], help="analyse one email using models from a previous run")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="email text")
    g.add_argument("--file", help="path to a text file containing the email")

    args = ap.parse_args(argv)

    if args.cmd == "list-stages":
        for n in ORDER:
            s = BY_NAME[n]
            print(f"{n:13s} {s.description}  [requires: {', '.join(s.requires) or '-'}]"
                  f"{'  (needs Gemini key)' if s.needs_llm else ''}")
        return 0

    cfg = build_cfg(args)
    try:
        if args.cmd == "run":
            Pipeline(cfg).run(args.stages, args.skip, args.continue_on_error)
        else:
            text = args.text if args.text else open(args.file, encoding="utf-8").read()
            analyze_text(cfg, text)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
