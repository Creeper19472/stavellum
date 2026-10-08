"""Actionable failures shared by input format adapters."""

from stavellum.domain.models import Diagnostic


class ImportFailure(ValueError):
    """The input cannot be represented safely; diagnostics accompany the failure."""

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or [Diagnostic("error", "import_failed", message)]
