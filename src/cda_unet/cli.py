"""Command-line entry points for prepare, train, evaluate and label-free inference."""
import argparse
import json
from pathlib import Path
from .config import load_config
from . import __version__


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cda-unet", description="CDA-UNet cytometric gating")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate subject CSVs and prepare compact training images")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--cache-dir", type=Path, required=True)
    prep.add_argument("--parents-dir", type=Path)
    train = commands.add_parser("train", help="Train CDA-UNet with subject-level held-out folds")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--fold", default="all", help="Zero-based fold index or all (sequential)")
    train.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    train.add_argument("--resume", type=Path, help="Resume a trusted local checkpoint for one fold")
    train.add_argument("--epochs", type=int, help="Override final training epoch (default: 1000 from YAML)")
    evaluate = commands.add_parser("evaluate", help="Evaluate only the checkpoint's held-out subjects")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--parents-dir", type=Path)
    predict = commands.add_parser("predict", help="Predict all gates for a CSV without reading annotations")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True, help="Output .npz with cell-level predictions")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--parents-file", type=Path)
    aggregate = commands.add_parser("aggregate", help="Combine held-out fold reports without duplicate subjects")
    aggregate.add_argument("--reports", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    from . import engine
    try:
        if args.command == "prepare":
            result = engine.prepare_data(load_config(args.config), args.data_dir, args.cache_dir, args.parents_dir)
            print(f"Prepared cache: {args.cache_dir}")
        elif args.command == "train":
            config = load_config(args.config)
            if args.fold == "all":
                if args.resume:
                    raise ValueError("--resume requires a single --fold")
                folds = range(config["training"]["folds"])
            else:
                try: folds = [int(args.fold)]
                except ValueError: raise ValueError("--fold must be an integer or all") from None
            for fold in folds:
                checkpoint = engine.train_fold(config, args.cache_dir, args.output_dir, fold,
                    device=args.device, resume=args.resume, epochs=args.epochs)
                print(f"Completed fold {fold}: {checkpoint}", flush=True)
        elif args.command == "evaluate":
            result = engine.evaluate_checkpoint(args.checkpoint, args.data_dir, args.output_dir,
                                                device=args.device, parents_dir=args.parents_dir)
            print(json.dumps(result.get("summary", result), indent=2))
        elif args.command == "predict":
            engine.predict_file(args.checkpoint, args.input, args.output, device=args.device,
                                parents_file=args.parents_file)
            print(f"Predictions: {args.output}")
        elif args.command == "aggregate":
            result = engine.aggregate_reports(args.reports, args.output_dir)
            print(json.dumps(result.get("summary", result), indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.exit(2, f"Error: {error}\n")
