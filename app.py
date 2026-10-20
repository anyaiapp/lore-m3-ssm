"""Entry point: launch the Lore-M3 Studio UI (or CLI chat with --cli)."""

import subprocess
import sys


def main() -> None:
    if "--cli" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--cli"]
        subprocess.run([sys.executable, "scripts/chat.py", *args], check=False)
        return
    subprocess.run([sys.executable, "scripts/ui.py", *sys.argv[1:]], check=False)


if __name__ == "__main__":
    main()
