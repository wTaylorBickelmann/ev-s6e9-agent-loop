"""Thin CLI: python -m ev_s6e9 <download|train|predict|submit|eda|build_site>."""

from __future__ import annotations

import argparse
import sys

from ev_s6e9.paths import DATA_RAW, EXPERIMENTS_MD, SUB_CSV, TRAIN_CSV
from ev_s6e9.schema import COMPETITION


def _parser() -> argparse.ArgumentParser:
    """Subcommands: download, train, predict, submit, eda, build_site."""

    p = argparse.ArgumentParser(prog="python -m ev_s6e9")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="kaggle competitions download into data/raw")
    d.add_argument("--synth", action="store_true", help="write schema-accurate fake CSVs instead")

    t = sub.add_parser("train", help="stratified CV + save OOF/models + append EXPERIMENTS.md")
    t.add_argument("--strategy", choices=["lgbm", "deotte", "ensemble", "catboost", "hgb", "rank_blend"], default="lgbm")
    t.add_argument("--oof-a", default=None, help="first OOF csv for ensemble/rank_blend (default exps/exp0026/oof.csv)")
    t.add_argument("--oof-b", default=None, help="second OOF csv for ensemble/rank_blend (default exps/exp0024/oof.csv)")
    t.add_argument("--oof-c", default=None, help="third OOF csv for rank_blend (default exps/exp0029/oof.csv)")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--n-estimators", type=int, default=None)
    t.add_argument("--note", default="", help="one-line takeaway for EXPERIMENTS.md")
    t.add_argument("--no-log", action="store_true", help="do not append EXPERIMENTS.md")
    t.add_argument("--synth", action="store_true", help="train on tiny synthetic data (no kaggle files)")
    t.add_argument("--freq", action="store_true", help="add value-count features (Annual_Income_USD, Daily_Commute_km)")
    t.add_argument("--te", action="store_true", help="fold-safe target encoding of Annual_Income_USD")
    t.add_argument("--seeds", default=None, help="comma-separated model seeds for multi-seed blend (e.g. 42,43,44)")
    t.add_argument("--weight-search", action="store_true", help="grid-search non-equal blend weights for m1/m2/m3")
    t.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="model param override (repeatable), e.g. --override max_depth=8")

    pr = sub.add_parser("predict", help="average fold models → outputs/submission.csv")
    pr.add_argument("--strategy", choices=["lgbm", "deotte"], default=None)

    s = sub.add_parser("submit", help="kaggle competitions submit")
    s.add_argument("-m", "--message", default="lgbm baseline")
    s.add_argument("-f", "--file", default=str(SUB_CSV))

    e = sub.add_parser("eda", help="write plots under reports/")
    e.add_argument("--synth", action="store_true")

    b = sub.add_parser("build_site", help="static HTML report gallery → _site/")
    b.add_argument("-o", "--out", default=None, help="output directory (default _site)")
    b.add_argument("--sample", action="store_true", help="force committed sample (skip data/raw)")
    return p


def _coerce(value: str):
    """Coerce a CLI override value to int/float when possible, else keep as str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _need_train_csv(synth: bool):
    """Load train.csv, or a tiny synth frame when `--synth` / file missing."""

    from ev_s6e9.data import load_train, synth as make_synth

    if synth:
        return make_synth(800, seed=0)
    if not TRAIN_CSV.exists():
        sys.exit(
            f"missing {TRAIN_CSV}\n"
            "  1. put kaggle.json in ~/.kaggle/ (chmod 600)\n"
            "  2. python -m ev_s6e9 download\n"
            "or: python -m ev_s6e9 train --synth"
        )
    return load_train()


def main(argv: list[str] | None = None) -> None:
    """Dispatch `python -m ev_s6e9 <cmd>` to the matching library entrypoint."""

    args = _parser().parse_args(argv)

    if args.cmd == "download":
        from ev_s6e9.data import download, write_synth_raw

        if args.synth:
            write_synth_raw()
            print(f"wrote synthetic CSVs under {DATA_RAW}")
        else:
            download()
            print(f"downloaded {COMPETITION} → {DATA_RAW}")
        return

    if args.cmd == "train":
        from ev_s6e9.viz import eda

        df = _need_train_csv(args.synth)
        ov = {}
        for kv in args.override:
            key, sep, val = kv.partition("=")
            if not key or not sep:
                sys.exit(f"bad --override {kv!r}; expected KEY=VALUE")
            ov[key] = _coerce(val)
        if args.n_estimators is not None:
            ov["n_estimators"] = args.n_estimators
        if args.synth:
            ov.setdefault("n_estimators", 40)
        if args.strategy == "ensemble":
            from ev_s6e9.deotte import train_ensemble

            train_ensemble(
                df,
                oof_a=args.oof_a,
                oof_b=args.oof_b,
                folds=args.folds,
                seed=args.seed,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
            )
            return
        if args.strategy == "rank_blend":
            from ev_s6e9.deotte import train_rank_blend

            train_rank_blend(
                df,
                oof_a=args.oof_a,
                oof_b=args.oof_b,
                oof_c=args.oof_c,
                folds=args.folds,
                seed=args.seed,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
            )
            return
        if args.strategy == "deotte":
            from ev_s6e9.data import load_test
            from ev_s6e9.deotte import train as train_deotte
            from ev_s6e9.paths import TEST_CSV

            test_df = load_test() if TEST_CSV.exists() else None
            seeds_list = None
            if args.seeds:
                seeds_list = [int(s) for s in args.seeds.split(",")]
            train_deotte(
                df,
                test_df,
                folds=args.folds,
                seed=args.seed,
                seeds=seeds_list,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
                model_overrides=ov or None,
                freq=args.freq,
                te=args.te,
                weight_search=args.weight_search,
            )
        elif args.strategy == "catboost":
            from ev_s6e9.data import load_test
            from ev_s6e9.paths import TEST_CSV
            from ev_s6e9.train import train_catboost

            test_df = load_test() if TEST_CSV.exists() else None
            train_catboost(
                df,
                folds=args.folds,
                seed=args.seed,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
                model_overrides=ov or None,
                freq=args.freq,
                te=args.te,
                test=test_df,
            )
        elif args.strategy == "hgb":
            from ev_s6e9.data import load_test
            from ev_s6e9.paths import TEST_CSV
            from ev_s6e9.train import train_hgb

            test_df = load_test() if TEST_CSV.exists() else None
            train_hgb(
                df,
                folds=args.folds,
                seed=args.seed,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
                model_overrides=ov or None,
                freq=args.freq,
                te=args.te,
                test=test_df,
            )
        else:
            from ev_s6e9.data import load_test
            from ev_s6e9.paths import TEST_CSV
            from ev_s6e9.train import train

            test_df = load_test() if TEST_CSV.exists() else None
            train(
                df,
                folds=args.folds,
                seed=args.seed,
                log=not args.no_log and not args.synth,
                note=args.note,
                experiments_path=EXPERIMENTS_MD,
                model_overrides=ov or None,
                freq=args.freq,
                te=args.te,
                test=test_df,
            )
        eda(df)
        return

    if args.cmd == "predict":
        from ev_s6e9.data import load_test
        from ev_s6e9.predict import predict
        from ev_s6e9.paths import TEST_CSV

        if not TEST_CSV.exists():
            sys.exit(f"missing {TEST_CSV}; run download (or download --synth)")
        predict(load_test(), strategy=args.strategy)
        return

    if args.cmd == "submit":
        from pathlib import Path

        from ev_s6e9.submit import submit

        submit(path=Path(args.file), message=args.message)
        return

    if args.cmd == "eda":
        from ev_s6e9.viz import eda

        df = _need_train_csv(args.synth)
        eda(df)
        return

    if args.cmd == "build_site":
        from pathlib import Path

        from ev_s6e9.site import build_site

        build_site(out=Path(args.out) if args.out else None, sample=args.sample)


if __name__ == "__main__":
    main()
