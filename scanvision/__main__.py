"""Allow ``python -m scanvision``."""

import sys

from scanvision.cli import main

if __name__ == "__main__":
    sys.exit(main())