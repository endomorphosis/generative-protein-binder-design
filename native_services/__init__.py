"""Host-native model service wrappers.

These services are intended to be run directly on an ARM64 host (e.g. DGX Spark)
inside the user's conda environments that have real AlphaFold2 / RFdiffusion
installed.

They expose a small NIM-compatible REST surface so the MCP server can route to
local-native inference without relying on CI-only shim containers.
"""
