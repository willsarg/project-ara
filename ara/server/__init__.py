# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The ``ara server`` coordinator — a Django app that is a plain HTTP *client* of node APIs.

Mode 2 of the node/server design: the server keeps a registry of nodes and shows/drives them
through the Django admin (the v1 dashboard); nodes don't know the server exists. Django is the
optional ``[server]`` extra, so everything here is import-light and only loaded by ``ara server``
— it never runs inside ARA's own process, which is why ``ara/server/*`` sits outside the coverage
gate (like ``ara/workers/*``). The CLI surface (``ara/cli.py:render_server``) IS in the gate and is
unit-tested against a mocked :mod:`ara.server.service`.
"""
