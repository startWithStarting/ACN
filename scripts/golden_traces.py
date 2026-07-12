#!/usr/bin/env python3
"""Golden-trace regression harness.

Runs small, seeded, headless scenarios through the real envs and compares
per-step agent positions/rewards against stored golden files. This is the
mechanical gate for the acceptance criterion "no existing simulations change
when communication is disabled" (docs/communication_implementation_plan.md).

Usage:
    uv run python scripts/golden_traces.py --capture   # (re)write goldens
    uv run python scripts/golden_traces.py --check     # diff current vs stored

Goldens live in tests/golden/*.json. They are machine-local (float ops may
differ across hardware); regenerate with --capture when setting up a new
machine, and only ever regenerate on a commit whose behavior change is
intended and reviewed.
"""

import argparse
import copy
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.agents.factory import create_agents_from_config  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "golden")
SEED = 42
CYCLES = 15
ROUND_DECIMALS = 6

# scenario name -> (base config, mode)
SCENARIOS = {
    "parallel_kinematic": ("config/avoidant_config.yaml", "parallel"),
    "parallel_physics_field": ("config/avoidant_physics_attractor_preview_config.yaml", "parallel"),
    "aec_kinematic": ("config/avoidant_config.yaml", "aec"),
}


def _tiny_config(base_path: str) -> dict:
    """Load a scenario config and shrink it to a fast, headless variant."""
    config = copy.deepcopy(load_config(os.path.join(REPO_ROOT, base_path)))
    env_cfg = config.setdefault("environment", {})
    env_cfg["max_cycles"] = CYCLES
    env_cfg["render_mode"] = None
    env_cfg["save_episode_gifs"] = False
    config.setdefault("analysis", {})["trace"] = {"enabled": False}
    for side in ("blue_agents", "red_agents"):
        specs = config.get("agents", {}).get(side, [])
        if specs:
            specs[0]["count"] = 3
            del specs[1:]
    return config


def _snapshot(env) -> dict:
    return {
        name: [round(float(obj.x), ROUND_DECIMALS), round(float(obj.y), ROUND_DECIMALS)]
        for name, obj in sorted(env.agent_objects.items())
    }


def _run_parallel(config: dict) -> list:
    from src.env.parallel_env import ParallelGameEnv

    env_cfg = dict(config.get("environment", {}))
    agents = create_agents_from_config(config.get("agents", {}), env_cfg)
    env = ParallelGameEnv(agents=agents, **env_cfg)
    observations, _infos = env.reset(seed=SEED)

    steps = []
    for _cycle in range(CYCLES):
        if not env.agents:
            break
        actions = {
            name: env.agent_objects[name].choose_action(observations[name])
            for name in env.agents
            if name in observations
        }
        observations, rewards, terminations, _truncations, _infos = env.step(actions)
        steps.append(
            {
                "positions": _snapshot(env),
                "rewards": {k: round(float(v), ROUND_DECIMALS) for k, v in sorted(rewards.items())},
                "terminations": dict(sorted(terminations.items())),
            }
        )
    env.close()
    return steps


def _run_aec(config: dict) -> list:
    from src.env.aec_env import AECGameEnv

    env_cfg = dict(config.get("environment", {}))
    agents = create_agents_from_config(config.get("agents", {}), env_cfg)
    env = AECGameEnv(agents=agents, **env_cfg)
    env.reset(seed=SEED)

    steps = []
    for _cycle in range(CYCLES):
        if not env.agents:
            break
        for _turn in list(env.agents):
            agent_name = env.agent_selection
            observation = env.observe(agent_name)
            if env.terminations[agent_name] or env.truncations[agent_name]:
                action = None
            else:
                action = env.agent_objects[agent_name].choose_action(observation)
            env.step(action)
        steps.append(
            {
                "positions": _snapshot(env),
                "rewards": {k: round(float(v), ROUND_DECIMALS) for k, v in sorted(env.rewards.items())},
                "terminations": dict(sorted(env.terminations.items())),
            }
        )
    env.close()
    return steps


def run_scenario(name: str) -> list:
    base_path, mode = SCENARIOS[name]
    config = _tiny_config(base_path)
    runner = _run_parallel if mode == "parallel" else _run_aec
    return runner(config)


def capture() -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in SCENARIOS:
        trace = run_scenario(name)
        path = os.path.join(GOLDEN_DIR, f"{name}.json")
        with open(path, "w") as handle:
            json.dump(trace, handle, indent=1, sort_keys=True)
        print(f"captured {name}: {len(trace)} steps -> {os.path.relpath(path, REPO_ROOT)}")


def check() -> int:
    failures = 0
    for name in SCENARIOS:
        path = os.path.join(GOLDEN_DIR, f"{name}.json")
        if not os.path.exists(path):
            print(f"MISSING golden for {name}; run --capture first")
            failures += 1
            continue
        with open(path) as handle:
            expected = json.load(handle)
        actual = json.loads(json.dumps(run_scenario(name)))  # normalize types
        if actual == expected:
            print(f"OK   {name} ({len(actual)} steps)")
        else:
            failures += 1
            print(f"FAIL {name}")
            for step_index, (exp, act) in enumerate(zip(expected, actual)):
                if exp != act:
                    print(f"  first divergence at step {step_index}")
                    for key in ("positions", "rewards", "terminations"):
                        if exp.get(key) != act.get(key):
                            print(f"    {key} expected: {exp.get(key)}")
                            print(f"    {key} actual:   {act.get(key)}")
                    break
            if len(expected) != len(actual):
                print(f"  step count: expected {len(expected)}, actual {len(actual)}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", action="store_true", help="write golden files")
    group.add_argument("--check", action="store_true", help="compare against goldens")
    args = parser.parse_args()

    if args.capture:
        capture()
    else:
        sys.exit(1 if check() else 0)


if __name__ == "__main__":
    main()
