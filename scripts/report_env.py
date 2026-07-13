#!/usr/bin/env python
"""Print the environment report + recommended compute plan.

Usage: python scripts/report_env.py
"""

from wildlife.utils.env_report import main

if __name__ == "__main__":
    raise SystemExit(main())
