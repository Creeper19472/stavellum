"""Shared errors for Vulkan graphics capability and resource failures."""


class GpuBackendError(RuntimeError):
    """A graphics capability/resource failure, distinct from a drawing error."""
