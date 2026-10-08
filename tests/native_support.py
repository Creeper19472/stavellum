"""Shared guards for tests that need a working native Vulkan device."""

from __future__ import annotations

from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def vulkan_device_available() -> bool:
    """Probe once per session whether the native DLL can open a device."""
    try:
        from stavellum.rendering._rhi import RhiTarget

        target = RhiTarget(4, 4, 0, "vulkan")
        target.close()
    except Exception:
        return False
    return True


def require_vulkan_device() -> None:
    if not vulkan_device_available():
        pytest.skip("No usable Vulkan device for the native renderer")
