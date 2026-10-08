"""Package the built native backends as a Windows x64 wheel."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Initial source installs must work before the development SDK is prepared.
        if version == "editable":
            return
        rhi_root = Path(self.root) / "src/stavellum/native/rhi"
        core_root = Path(self.root) / "src/stavellum/native/core"
        names = [("stavellum_rust.dll",) if (rhi_root / "stavellum_rust.dll").is_file()
                 else ("stavellum_rhi.dll", "quad.vert.qsb", "quad.frag.qsb")][0]
        missing = [name for name in names if not (rhi_root / name).is_file()]
        if missing:
            raise RuntimeError(
                "Build and install the Vulkan backend before packaging: "
                "uv run python scripts/build_rust.py --install; missing " + ", ".join(missing)
            )
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-win_amd64"
        for name in names:
            build_data["force_include"][str(rhi_root / name)] = (
                "stavellum/native/rhi/" + name
            )
        core = core_root / "stavellum_core.dll"
        if core.is_file():
            # Optional accelerator: the Python fallback keeps the wheel
            # functional when the scene core DLL is absent.
            build_data["force_include"][str(core)] = "stavellum/native/core/stavellum_core.dll"
