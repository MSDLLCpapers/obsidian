#!/usr/bin/env python3
"""
Smart dependency generator for Docker containers.

This script:
1. Parses obsidian's pyproject.toml for core dependencies + API/LLM extras
2. Reads current conda environment for installed versions
3. Generates requirements_docker.txt with pinned versions
4. Handles CPU-only PyTorch variants
5. Creates a report showing all decisions
"""

import argparse
import json
import subprocess as sp
import sys
from pathlib import Path

try:
    import toml
except ImportError:
    print("Error: toml package not found. Install with: pip install toml")
    sys.exit(1)


class DependencyGenerator:
    # PyTorch ecosystem packages that need CPU-only index
    TORCH_ECOSYSTEM_PACKAGES = {
        "torch",
        "botorch",
        "gpytorch",
        "torchvision",
        "torchaudio",
    }

    def __init__(
        self,
        obsidian_path: Path,
        strategy: str = "pin-current",
        python_version: str | None = None,
        conda_env: str = "obsidian",
    ):
        self.obsidian_path = obsidian_path
        self.strategy = strategy
        self.python_version = python_version
        self.conda_env = conda_env
        self.conda_packages: dict[str, str] = {}
        self.decisions: list[dict] = []

    def load_obsidian_dependencies(self) -> set[str]:
        """Parse obsidian's pyproject.toml for core dependencies + API/LLM extras."""
        pyproject_path = self.obsidian_path / "pyproject.toml"

        if not pyproject_path.exists():
            raise FileNotFoundError(f"Cannot find {pyproject_path}")

        print(f"📖 Reading dependencies from {pyproject_path}")

        with open(pyproject_path, "r") as f:
            data = toml.load(f)

        # Get main dependencies (not dev/extras)
        dependencies = data.get("tool", {}).get("poetry", {}).get("dependencies", {})

        # Remove python from dependencies (handled separately)
        if "python" in dependencies:
            del dependencies["python"]

        # Get API/LLM extras for server functionality
        extras = data.get("tool", {}).get("poetry", {}).get("extras", {})
        llm_extras = set(extras.get("llm", []))

        print(f"   📡 Including API/LLM extras: {', '.join(sorted(llm_extras))}")

        # Include non-optional dependencies + API/LLM extras
        core_dependencies = {}
        for dep, spec in dependencies.items():
            # If spec is a dict with "optional": true, only include if it's in API/LLM extras
            if isinstance(spec, dict) and spec.get("optional", False):
                if dep in llm_extras:
                    print(f"   ➕ Including API/LLM dependency: {dep}")
                    core_dependencies[dep] = spec
                else:
                    print(f"   ⏭️  Skipping optional dependency: {dep}")
                continue
            core_dependencies[dep] = spec

        dep_set = set(core_dependencies.keys())

        print(f"   Found {len(dep_set)} dependencies (core + API/LLM extras)")
        return dep_set

    def load_dev_dependencies(self) -> set[str]:
        """Parse obsidian's pyproject.toml for dev extras."""
        pyproject_path = self.obsidian_path / "pyproject.toml"

        if not pyproject_path.exists():
            raise FileNotFoundError(f"Cannot find {pyproject_path}")

        print(f"\n📖 Reading dev dependencies from {pyproject_path}")

        with open(pyproject_path, "r") as f:
            data = toml.load(f)

        # Get dev extras
        extras = data.get("tool", {}).get("poetry", {}).get("extras", {})
        dev_extras = set(extras.get("dev", []))

        print(f"   🔧 Found {len(dev_extras)} dev dependencies")
        for dep in sorted(dev_extras):
            print(f"      - {dep}")

        return dev_extras

    def get_conda_packages(self) -> dict[str, str]:
        """Get installed package versions from conda environment."""
        print(f"📦 Reading conda environment: {self.conda_env}")

        try:
            # Try to get package list
            result = sp.run(
                ["conda", "list", "-n", self.conda_env, "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            packages = json.loads(result.stdout)

            # Create name -> version mapping
            pkg_dict = {}
            for pkg in packages:
                name = pkg["name"].lower().replace("_", "-")  # Normalize name
                version = pkg["version"]
                pkg_dict[name] = version

            print(f"   Found {len(pkg_dict)} installed packages")
            return pkg_dict

        except sp.CalledProcessError as e:
            print(f"❌ Error reading conda environment: {e}")
            print("   Falling back to pip list...")
            return self._get_pip_packages()

    def _get_pip_packages(self) -> dict[str, str]:
        """Fallback: Get packages from pip."""
        try:
            result = sp.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            packages = json.loads(result.stdout)

            pkg_dict = {}
            for pkg in packages:
                name = pkg["name"].lower()
                version = pkg["version"]
                pkg_dict[name] = version

            return pkg_dict

        except Exception as e:
            print(f"❌ Error reading pip packages: {e}")
            return {}

    def get_current_python_version(self) -> str:
        """Get Python version from conda environment."""
        if self.python_version:
            return self.python_version

        python_ver = self.conda_packages.get("python", None)
        if python_ver:
            print(f"   Detected Python {python_ver} in conda environment")
            return python_ver

        # Fallback to system Python
        import platform

        python_ver = platform.python_version()
        print(f"   Using system Python {python_ver}")
        return python_ver

    def resolve_version(self, package: str) -> tuple[str | None, str]:
        """
        Resolve version for a package based on strategy.

        Returns: (version, reason)
        """
        # Special handling for torch ecosystem
        if package in self.TORCH_ECOSYSTEM_PACKAGES:
            return self._resolve_torch_version(package)

        # Get current version from conda
        current_version = self.conda_packages.get(package.lower(), None)

        if current_version is None:
            # Try with underscores
            current_version = self.conda_packages.get(
                package.lower().replace("-", "_"), None
            )

        if current_version is None:
            return None, f"Not found in conda environment"

        if self.strategy == "pin-current":
            return current_version, "Pinned to current conda version"

        elif self.strategy == "upgrade-patch":
            # For now, just use current version
            # In a full implementation, we'd query PyPI for latest patch
            return (
                current_version,
                "Using current version (upgrade-patch not fully implemented)",
            )

        return current_version, "Default"

    def _resolve_torch_version(self, package: str) -> tuple[str | None, str]:
        """Special handling for PyTorch packages - use base version without suffix."""
        current_version = self.conda_packages.get(package, None)

        if current_version is None:
            return None, "Not found in conda environment"

        # Remove any existing +cpu or +cu118 suffixes
        base_version = current_version.split("+")[0]

        # Don't add +cpu suffix - let the index-url handle CPU variant selection
        reason = f"CPU-only variant of {current_version} (via index-url)"

        return base_version, reason

    def generate_requirements(
        self, output_path: Path, dev_output_path: Path | None = None, report_path: Path | None = None
    ) -> None:
        """Generate requirements_docker.txt and requirements_docker_dev.txt files."""
        print("\n🔧 Generating requirements files...")

        # Load dependencies
        obsidian_deps = self.load_obsidian_dependencies()
        dev_deps = self.load_dev_dependencies() if dev_output_path else set()
        self.conda_packages = self.get_conda_packages()
        python_version = self.get_current_python_version()

        # Resolve versions for all dependencies
        requirements = []
        torch_ecosystem = []

        for dep in sorted(obsidian_deps):
            version, reason = self.resolve_version(dep)

            if version:
                if dep in self.TORCH_ECOSYSTEM_PACKAGES:
                    torch_ecosystem.append((dep, version))
                else:
                    requirements.append((dep, version))

                self.decisions.append(
                    {"package": dep, "version": version, "reason": reason}
                )
            else:
                print(f"   ⚠️  Skipping {dep}: {reason}")
                self.decisions.append(
                    {"package": dep, "version": None, "reason": reason, "skipped": True}
                )

        # Write base requirements file
        print(f"\n📝 Writing {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Header
            f.write("# Generated requirements for Docker container\n")
            f.write(f"# Strategy: {self.strategy}\n")
            f.write(f"# Python: {python_version}\n")
            f.write("# Source: obsidian core dependencies + API/LLM extras + conda environment\n")
            f.write(f"# Generated: {sp.check_output(['date'], text=True).strip()}\n\n")

            # PyTorch ecosystem (with CPU-only index)
            if torch_ecosystem:
                f.write("# PyTorch ecosystem (CPU-only)\n")
                f.write(
                    "# Keep CPU index for torch ecosystem to avoid CUDA dependencies\n"
                )
                f.write("--index-url https://download.pytorch.org/whl/cpu\n")
                f.write("--extra-index-url https://pypi.org/simple\n")
                f.write("# SSL certificate workarounds for corporate environments\n")
                f.write("--trusted-host download.pytorch.org\n")
                f.write("--trusted-host pypi.org\n")
                f.write("--trusted-host files.pythonhosted.org\n")
                f.write("\n")
                for pkg, ver in torch_ecosystem:
                    f.write(f"{pkg}=={ver}\n")
                f.write("\n")

            # Other dependencies (will use extra-index-url from above)
            if requirements:
                f.write("# Other core dependencies\n")
                for pkg, ver in requirements:
                    f.write(f"{pkg}=={ver}\n")

        print(
            f"   ✅ Generated {len(requirements) + len(torch_ecosystem)} package specifications"
        )

        # Write dev requirements file
        if dev_output_path and dev_deps:
            print(f"\n📝 Writing {dev_output_path}")
            dev_requirements = []

            for dep in sorted(dev_deps):
                version, reason = self.resolve_version(dep)
                if version:
                    dev_requirements.append((dep, version))
                    self.decisions.append(
                        {"package": dep, "version": version, "reason": reason, "context": "dev"}
                    )
                else:
                    print(f"   ⚠️  Skipping dev dependency {dep}: {reason}")

            with open(dev_output_path, "w") as f:
                f.write("# Generated dev requirements for Docker dev container\n")
                f.write(f"# Strategy: {self.strategy}\n")
                f.write(f"# Python: {python_version}\n")
                f.write("# Source: obsidian dev extras + conda environment\n")
                f.write("# Note: Dev container extends base image (core + API/LLM deps already included)\n")
                f.write(f"# Generated: {sp.check_output(['date'], text=True).strip()}\n\n")
                f.write("# Development and testing dependencies (added to base)\n")
                for pkg, ver in dev_requirements:
                    f.write(f"{pkg}=={ver}\n")

            print(f"   ✅ Generated {len(dev_requirements)} dev package specifications")

        # Write report
        if report_path:
            print(f"\n📊 Writing report to {report_path}")
            with open(report_path, "w") as f:
                json.dump(
                    {
                        "strategy": self.strategy,
                        "python_version": python_version,
                        "total_packages": len(self.decisions),
                        "decisions": self.decisions,
                    },
                    f,
                    indent=2,
                )
            print(f"   ✅ Report written")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Docker requirements from obsidian dependencies"
    )
    parser.add_argument(
        "--obsidian-path",
        type=Path,
        default=None,
        help="Path to obsidian repository (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--strategy",
        choices=["pin-current", "upgrade-patch"],
        default="pin-current",
        help="Version pinning strategy (default: pin-current)",
    )
    parser.add_argument(
        "--python-version", help="Python version to use (default: detect from conda)"
    )
    parser.add_argument(
        "--conda-env",
        default="obsidian",
        help="Conda environment name (default: obsidian)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for requirements file (default: <obsidian-root>/container/docker/requirements_docker.txt)",
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=None,
        help="Output path for dev requirements file (default: <obsidian-root>/container/docker/requirements_docker_dev.txt)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output path for generation report (default: <obsidian-root>/container/requirements_report.json)",
    )

    args = parser.parse_args()

    # Auto-detect obsidian path from script location if not provided
    if args.obsidian_path is None:
        # Script is in container/generate_requirements.py, so parent is obsidian root
        script_dir = Path(__file__).parent
        args.obsidian_path = script_dir.parent
        print(f"🔍 Auto-detected obsidian root: {args.obsidian_path}")

    # Resolve to absolute path
    args.obsidian_path = args.obsidian_path.resolve()

    # Validate obsidian path
    if not args.obsidian_path.exists():
        print(f"❌ Error: Obsidian path not found: {args.obsidian_path}")
        print(f"   Please specify correct path with --obsidian-path")
        sys.exit(1)

    # Set default output paths relative to obsidian root
    if args.output is None:
        args.output = args.obsidian_path / "container" / "docker" / "requirements_docker.txt"
    if args.dev_output is None:
        args.dev_output = args.obsidian_path / "container" / "docker" / "requirements_docker_dev.txt"
    if args.report is None:
        args.report = args.obsidian_path / "container" / "requirements_report.json"

    print("=" * 70)
    print("🐳 Docker Requirements Generator")
    print("=" * 70)

    generator = DependencyGenerator(
        obsidian_path=args.obsidian_path,
        strategy=args.strategy,
        python_version=args.python_version,
        conda_env=args.conda_env,
    )

    try:
        generator.generate_requirements(
            output_path=args.output, dev_output_path=args.dev_output, report_path=args.report
        )

        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\n📄 Base requirements: {args.output}")
        print(f"📄 Dev requirements:  {args.dev_output}")
        print(f"📊 Report:            {args.report}")
        print("\nNext steps:")
        print(f"  1. Review {args.output} and {args.dev_output}")
        print(
            f"  2. Build Docker image: cd container/scripts && ./build_docker.sh base"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
