"""Entry point: python -m context_segment_value [demo|evaluate]."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
