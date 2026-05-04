#!/usr/bin/env python3
"""Compatibility entrypoint.

`main_q.py` is now the canonical entrypoint. This shim preserves old
`python main.py --config ...` commands while routing to `main_q`.
"""
from __future__ import annotations

from main_q import main


if __name__ == "__main__":
    main()

