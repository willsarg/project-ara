# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Run ARA through the same blessed wrapper as the installed ``ara`` script."""
from __future__ import annotations

import sys

from ara.cli import main


sys.exit(main())
