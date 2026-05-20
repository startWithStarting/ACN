import ast
import unittest
from pathlib import Path


class TestArchitectureBoundaries(unittest.TestCase):
    def test_simulation_core_does_not_import_api_or_storage_directly(self):
        """Keep DB/API infrastructure behind the recorder factory."""
        repo_root = Path(__file__).resolve().parents[1]
        allowed_storage_imports = {
            ("src/utils/history.py", "src.storage.history"),
        }
        core_paths = [
            repo_root / "main.py",
            repo_root / "main_parallel.py",
            repo_root / "run.py",
        ]
        for package in (
            "agents",
            "benchmark",
            "communication",
            "env",
            "physics",
            "training",
            "utils",
        ):
            package_dir = repo_root / "src" / package
            if package_dir.exists():
                core_paths.extend(package_dir.rglob("*.py"))

        violations = []
        for path in sorted(core_paths):
            rel_path = str(path.relative_to(repo_root))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in {"src.api", "src.storage"} or alias.name.startswith(
                            ("src.api.", "src.storage.")
                        ):
                            if (rel_path, alias.name) not in allowed_storage_imports:
                                violations.append(f"{rel_path} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in {"src.api", "src.storage"} or module.startswith(
                        ("src.api.", "src.storage.")
                    ):
                        if (rel_path, module) not in allowed_storage_imports:
                            violations.append(f"{rel_path} imports from {module}")
                    if module == "src":
                        for alias in node.names:
                            if alias.name in {"api", "storage"}:
                                violations.append(f"{rel_path} imports {alias.name} from src")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
