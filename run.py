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
import warnings
from dotenv import load_dotenv

load_dotenv()

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, configure_logging

logger = get_logger("acn.run")


def run_aec(config_path: str) -> None:
    """Run AEC (alternating turn) environment simulation."""
    import main as aec_module
    logger.info("Running AEC mode with config: {}", config_path)
    aec_module.main(config_path)


def run_parallel(config_path: str) -> None:
    """Run parallel environment simulation."""
    import main_parallel as parallel_module
    logger.info("Running parallel mode with config: {}", config_path)
    parallel_module.main(config_path)


def run_train(config_path: str) -> None:
    """Run training mode with PPO."""
    import main as aec_module
    logger.info("Running training mode with config: {}", config_path)
    aec_module.main(config_path, train_mode=True)


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
        run_aec(args.config)
    elif args.mode == "parallel":
        run_parallel(args.config)
    elif args.mode == "train":
        run_train(args.config)
    else:
        logger.error("Unknown mode: {}", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
