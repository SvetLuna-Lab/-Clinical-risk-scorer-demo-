# src/cli.py
"""
Command-line interface for the Clinical Risk Scorer demo.

Available commands:
- generate-data  — generate synthetic patient data (train/test CSVs)
- train          — train model and save artifacts
- evaluate       — evaluate a saved model on the test set
"""

import argparse
from pathlib import Path

from src.data.generate_synthetic import generate_synthetic_patients
from src.models.train import train_and_evaluate
from src.models.evaluate import evaluate_model
from src.paths import PROCESSED_DATA_DIR, PROJECT_ROOT


def cmd_generate_data(args: argparse.Namespace) -> None:
    """Generate synthetic train/test datasets and save them to data/processed/."""
    df = generate_synthetic_patients(n_samples=args.n_samples, seed=args.seed)

    # Simple shuffled train/test split
    df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(len(df_shuffled) * args.train_ratio)

    train_df = df_shuffled.iloc[:n_train].copy()
    test_df = df_shuffled.iloc[n_train:].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Generated train dataset: {train_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Generated test dataset:  {test_path.relative_to(PROJECT_ROOT)}")
    print(f"Train size: {len(train_df)}, test size: {len(test_df)}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train and evaluate the model defined in the config."""
    config_path = Path(args.config)
    metrics = train_and_evaluate(config_path)
    print("[OK] Training finished. Metrics:", metrics)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate an already trained model on the test set."""
    config_path = Path(args.config)
    model_path = Path(args.model_path)
    output_path = Path(args.output)

    metrics = evaluate_model(
        config_path=config_path,
        model_path=model_path,
        output_path=output_path,
    )
    print("[OK] Evaluation finished. Metrics:", metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clinical Risk Scorer demo CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-data
    p_gen = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic train/test datasets",
    )
    p_gen.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of synthetic patients to generate (default: 2000).",
    )
    p_gen.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    p_gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p_gen.set_defaults(func=cmd_generate_data)

    # train
    p_train = subparsers.add_parser(
        "train",
        help="Train model and save artifacts",
    )
    p_train.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Config file name or path (default: default.yaml in configs/).",
    )
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model on the test set",
    )
    p_eval.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Config file name or path (default: default.yaml in configs/).",
    )
    p_eval.add_argument(
        "--model-path",
        type=str,
        default="model_logistic_regression.joblib",
        help="Model file name or path (default: model_logistic_regression.joblib in models/).",
    )
    p_eval.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Output JSON file name for metrics (default: metrics.json in metrics/).",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
