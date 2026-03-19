"""OmniProof CLI — thin wrapper around the demo."""

from __future__ import annotations

import sys


def main() -> None:
    """Run the OmniProof demo."""
    # Ensure examples/data is findable
    from pathlib import Path

    # Add project root to path so demo imports work
    project_root = Path(__file__).resolve().parent.parent.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        print("Error: examples/ directory not found.", file=sys.stderr)
        print("The CLI demo requires the full repository checkout.", file=sys.stderr)
        sys.exit(1)

    # Import and run demo
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(examples_dir.parent))

    from examples.demo import main as demo_main

    demo_main()


if __name__ == "__main__":
    main()
