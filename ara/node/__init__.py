# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ARA Node — the headless push-only daemon: a client that enrolls with a coordinator, pulls work
over an outbound long-poll, and runs each job by reusing ARA's own CLI verbs (see ara/node/agent.py).
"""
