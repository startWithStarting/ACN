#!/usr/bin/env python3
"""Unified entry point for ACN experiments.

Supports three modes:
    aec     - AEC (alternating turn) environment simulation
    parallel - Parallel environment simulation
    train   - Training mode using PPO

Usage:
    python run.py --config config/avoidant_config.yaml --mode aec
    python run.py --config config/avoidant_config.yaml --mode parallel
    python run.py --config config/avoidant_config.yaml --mode train
"""

import argparse
import os
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, configure_logging

logger = get_logger("acn.run")


def run_aec(config_path: str, persist: bool = False, database_url: Optional[str] = None) -> None:
    """Run AEC (alternating turn) environment simulation."""
    import main as aec_module
    logger.info("Running AEC mode with config: {}", config_path)
    aec_module.main(config_path, persist=persist, database_url=database_url)


def run_parallel(config_path: str, persist: bool = False, database_url: Optional[str] = None) -> None:
    """Run parallel environment simulation."""
    import main_parallel as parallel_module
    logger.info("Running parallel mode with config: {}", config_path)
    parallel_module.main(config_path, persist=persist, database_url=database_url)


def run_train(
    config_path: str,
    persist: bool = False,
    database_url: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    """Run training mode.

    Routes on ``training.backend``: ``"marl"`` selects the TorchRL-backed
    team-selection trainer (``src.training.marl``); an absent or different
    backend keeps the legacy SB3 path unchanged (deprecated for new work).
    """
    config = load_config(config_path) or {}
    backend = (config.get("training") or {}).get("backend")
    if backend == "marl":
        from src.training.marl.runner import run_marl_training
        if persist:
            logger.warning(
                "--persist is ignored by the MARL trainer; training artifacts are "
                "file-backed under results/."
            )
        logger.info("Running MARL training mode with config: {}", config_path)
        run_marl_training(config_path, resume=resume)
        return
    import main as aec_module
    if resume:
        logger.warning("--resume is only supported by the MARL backend; ignoring it.")
    logger.info("Running legacy (SB3) training mode with config: {}", config_path)
    aec_module.main(config_path, train_mode=True, persist=persist, database_url=database_url)


def main():
    parser = argparse.ArgumentParser(
        description="ACN: Multi-agent simulation for researching communication and coordination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("ACN_CONFIG_PATH", "config/avoidant_config.yaml"),
        help="Path to the experiment configuration file (YAML)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aec", "parallel", "train"],
        default="parallel",
        help="Execution mode: aec (alternating turn), parallel (simultaneous), or train (PPO training)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Console logging level."
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist run history directly to Postgres instead of writing JSON trace files."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("ACN_DATABASE_URL"),
        help="Postgres URL for --persist. Defaults to ACN_DATABASE_URL."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint file to resume MARL training from (train mode, backend 'marl')."
    )

    args = parser.parse_args()

    # Configure logging early
    config = load_config(args.config)
    if config:
        logging_config = config.get("logging", {})
        configure_logging(logging_config)
    else:
        configure_logging({"level": args.log_level})

    # Dispatch to the appropriate runner
    if args.mode == "aec":
        run_aec(args.config, persist=args.persist, database_url=args.database_url)
    elif args.mode == "parallel":
        run_parallel(args.config, persist=args.persist, database_url=args.database_url)
    elif args.mode == "train":
        run_train(
            args.config,
            persist=args.persist,
            database_url=args.database_url,
            resume=args.resume,
        )
    else:
        logger.error("Unknown mode: {}", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
