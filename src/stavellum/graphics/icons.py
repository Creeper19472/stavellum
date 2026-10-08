"""Best-effort instrument aliases for cached, bundled SVG pictograms."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QImage, QImageReader, QPainter
from PySide6.QtSvg import QSvgRenderer

from stavellum.domain.models import IconAsset

from .qt import ensure_app

IMAGE_FILE_FILTER = "图标图片 (*.svg *.png *.jpg *.jpeg *.webp)"
FONT_AWESOME_VERSION = "7.3.1"

ICON_RESOURCES = {
    "bell": "bell-thin-100.svg",
    "drum": "drum-thin-100.svg",
    "piano": "piano-thin-100.svg",
    "keyboard": "piano-keyboard-thin-100.svg",
    "violin": "violin-thin-100.svg",
}
ICON_ALIASES = {
    "bell": "bell", "bells": "bell", "celesta": "bell",
    "glockenspiel": "bell", "music box": "bell", "tubular bells": "bell",
    "铃": "bell", "钟": "bell",
    "drum": "drum", "drums": "drum", "percussion": "drum",
    "timpani": "drum", "snare": "drum", "kick": "drum", "鼓": "drum", "打击乐": "drum",
    "piano": "piano", "grand piano": "piano", "钢琴": "piano",
    "keyboard": "keyboard", "piano keyboard": "keyboard", "electric piano": "keyboard",
    "harpsichord": "keyboard", "organ": "keyboard", "synth": "keyboard",
    "synthesizer": "keyboard", "键盘": "keyboard", "风琴": "keyboard",
    "violin": "violin", "viola": "violin", "cello": "violin",
    "double bass": "violin", "strings": "violin", "string": "violin",
    "小提琴": "violin", "中提琴": "violin", "大提琴": "violin",
    "低音提琴": "violin", "弦乐": "violin",
}


def _resource_name(kind: str) -> str | None:
    normalized = re.sub(r"[\s_-]+", " ", kind.strip().casefold().removesuffix(".svg"))
    for filename in ICON_RESOURCES.values():
        if normalized == re.sub(r"[\s_-]+", " ", filename.removesuffix(".svg")):
            return filename
    for alias in sorted(ICON_ALIASES, key=len, reverse=True):
        # English tokens avoid matching bassoon to bass. Chinese aliases may
        # appear within a section name such as 第一小提琴 or 打击乐组.
        if f" {alias} " in f" {normalized} " or (not alias.isascii() and alias in normalized):
            return ICON_RESOURCES[ICON_ALIASES[alias]]
    return None


@lru_cache(maxsize=len(ICON_RESOURCES))
def _load_icon(filename: str) -> QSvgRenderer | None:
    try:
        data = files("stavellum").joinpath("icons", filename).read_bytes()
        renderer = QSvgRenderer(data)
        if not renderer.isValid():
            return None
        renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        return renderer
    except (OSError, ValueError, RuntimeError):
        return None


def icon_renderer(kind: str) -> QSvgRenderer | None:
    """Return a reusable vector renderer, or None for the caller's fallback."""
    if kind.startswith("fa:"):
        ensure_app()
        return _decode_icon(fontawesome_svg(kind), "image/svg+xml")
    filename = _resource_name(kind)
    if filename is None:
        return None
    ensure_app()
    return _load_icon(filename)


@dataclass(frozen=True, slots=True)
class IconEntry:
    name: str
    style: str
    source: str
    path: str
    bundled: bool = False

    def asset(self) -> tuple[str, IconAsset]:
        if self.bundled:
            data = fontawesome_svg(f"fa:{self.style}:{self.name}")
        else:
            try:
                data = whiten_fontawesome(Path(self.path).read_bytes())
            except OSError as exc:
                raise ValueError(f"无法读取图标 {self.name}：{exc}") from exc
        return make_icon_asset(data, self.name, "image/svg+xml",
                               f"fontawesome:{self.style}:{self.name}")


@lru_cache(maxsize=1)
def free_catalog() -> tuple[IconEntry, ...]:
    index = json.loads(files("stavellum").joinpath("assets", "fontawesome", "index.json").read_text(encoding="utf-8"))
    return tuple(IconEntry(item["name"], item["style"], f"Free {FONT_AWESOME_VERSION}",
                           item["path"], True) for item in index["icons"])


@lru_cache(maxsize=1)
def _free_archive() -> zipfile.ZipFile:
    data = files("stavellum").joinpath("assets", "fontawesome", "icons.zip").read_bytes()
    return zipfile.ZipFile(io.BytesIO(data))


@lru_cache(maxsize=128)
def fontawesome_svg(reference: str) -> bytes:
    match = re.fullmatch(r"fa:(solid|regular|brands):([a-z0-9]+(?:-[a-z0-9]+)*)", reference)
    if match is None:
        raise ValueError(f"Font Awesome 图标名称无效：{reference}")
    style, name = match.groups()
    try:
        return whiten_fontawesome(_free_archive().read(f"{style}/{name}.svg"))
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"内置 Font Awesome Free 中没有图标 {style}/{name}。") from exc


def whiten_fontawesome(data: bytes) -> bytes:
    """Make standalone FA paths white while preserving attribution and opacity."""
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.fromstring(data, parser=parser)
    except ET.ParseError as exc:
        raise ValueError("Font Awesome SVG 数据无效。") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("图标必须为 SVG 文档。")
    root.set("color", "#ffffff")
    root.set("fill", "#ffffff")
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        # Some Pro downloads rely on CSS variables normally supplied by JS/CSS.
        for attr in ("fill", "stroke"):
            value = node.get(attr)
            if value and value != "none":
                node.set(attr, "#ffffff")
        style = node.get("style", "")
        if style:
            style = re.sub(r"(?<![\w-])(fill|stroke|color)\s*:\s*[^;]+",
                           lambda m: f"{m[1]}:none" if m[0].split(":", 1)[1].strip() == "none"
                           else f"{m[1]}:#ffffff", style)
            style = re.sub(r"var\(--fa-secondary-opacity(?:,\s*[^)]+)?\)", "0.4", style)
            style = re.sub(r"var\(--fa-primary-opacity(?:,\s*[^)]+)?\)", "1", style)
            node.set("style", style)
        if "fa-secondary" in node.get("class", "").split() and not node.get("opacity") and "opacity" not in style:
            node.set("opacity", "0.4")
        if node.tag.rsplit("}", 1)[-1] == "style" and node.text:
            node.text = re.sub(r"(fill|stroke|color)\s*:\s*[^;}]+", r"\1:#ffffff", node.text)
            node.text = re.sub(r"var\(--fa-secondary-opacity(?:,\s*[^)]+)?\)", "0.4", node.text)
            node.text = re.sub(r"var\(--fa-primary-opacity(?:,\s*[^)]+)?\)", "1", node.text)
    return ET.tostring(root, encoding="utf-8")


def scan_fontawesome(directory: str, cancelled: Callable[[], bool] = lambda: False) -> list[IconEntry]:
    """Index filenames only; full-canvas SVGs win over cropped duplicates."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Font Awesome 图标库目录不存在：{directory}")
    if root.name in ("svgs", "svgs-full"):
        roots = [root.parent / "svgs", root.parent / "svgs-full"]
    elif (root / "svgs").is_dir() or (root / "svgs-full").is_dir():
        roots = [root / "svgs", root / "svgs-full"]
    else:
        roots = [root]
    entries = {}
    def scan_error(error):
        raise ValueError(f"无法扫描图标库：{error}") from error
    for svg_root in roots:
        if not svg_root.is_dir():
            continue
        for folder, _, filenames in os.walk(svg_root, onerror=scan_error):
            if cancelled():
                return []
            for filename in sorted(filenames):
                if cancelled():
                    return []
                if Path(filename).suffix.lower() != ".svg":
                    continue
                path = Path(folder) / filename
                style = "-".join(path.parent.relative_to(svg_root).parts) or "custom"
                entries[(style, path.stem)] = IconEntry(path.stem, style, str(root), str(path))
    if not entries and not cancelled():
        raise ValueError("此目录没有可用的 SVG 图标。请选择已解压的 Font Awesome 图标库。")
    return sorted(entries.values(), key=lambda item: (item.style, item.name))


@lru_cache(maxsize=128)
def _decode_icon(data: bytes, media_type: str) -> QSvgRenderer | QImage:
    ensure_app()
    if media_type == "image/svg+xml":
        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            raise ValueError("SVG 图标数据无效。") from exc
        if root.tag.rsplit("}", 1)[-1] != "svg":
            raise ValueError("图标必须为 SVG 文档。")
        for node in root.iter():
            for attr, value in node.attrib.items():
                if attr.rsplit("}", 1)[-1] == "href" and not value.startswith(("#", "data:")):
                    raise ValueError("SVG 图标引用了外部资源。请先将图片嵌入 SVG 文件。")
        renderer = QSvgRenderer(data)
        if not renderer.isValid() or renderer.viewBoxF().isEmpty():
            raise ValueError("SVG 图标无效或画布为空。")
        renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        renderer.setAnimationEnabled(False)
        return renderer
    buffer = QBuffer()
    buffer.setData(QByteArray(data))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        raise ValueError("无法解码图标图片。")
    return image


def make_icon_asset(data: bytes, name: str, media_type: str, source: str = "custom") -> tuple[str, IconAsset]:
    _decode_icon(data, media_type)
    digest = hashlib.sha256(data).hexdigest()
    return f"asset:{digest}", IconAsset(name, media_type, base64.b64encode(data).decode("ascii"), source)


def import_icon(path: str | Path) -> tuple[str, IconAsset]:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取图标图片：{exc}") from exc
    if path.suffix.lower() == ".svg":
        media_type = "image/svg+xml"
    else:
        buffer = QBuffer()
        buffer.setData(QByteArray(data))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        image_format = bytes(QImageReader(buffer).format()).decode("ascii")
        media_type = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(image_format)
        if media_type is None:
            raise ValueError("图标仅支持 SVG、PNG、JPEG 和 WebP。")
    return make_icon_asset(data, path.name, media_type)


@lru_cache(maxsize=128)
def _embedded_icon(digest: str, encoded: str, media_type: str) -> QSvgRenderer | QImage:
    data = IconAsset("嵌入图标", media_type, encoded).decoded()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("嵌入图标的内容校验失败。")
    return _decode_icon(data, media_type)


def resolve_icon(kind: str, assets: dict[str, IconAsset] | None = None) -> QSvgRenderer | QImage | None:
    if not kind or kind in ("none", "unknown"):
        return None
    if kind.startswith("asset:"):
        asset = (assets or {}).get(kind[6:])
        if asset is None:
            raise ValueError("项目缺少所选的图标资源。请重新选择图标。")
        return _embedded_icon(kind[6:], asset.data, asset.media_type)
    return icon_renderer(kind)


def draw_image_icon(painter: QPainter, icon: QSvgRenderer | QImage, target: QRectF) -> None:
    if isinstance(icon, QSvgRenderer):
        icon.render(painter, target)
    else:
        size = icon.size().toSizeF()
        size.scale(target.size(), Qt.AspectRatioMode.KeepAspectRatio)
        rect = QRectF(target.center().x() - size.width() / 2,
                      target.center().y() - size.height() / 2, size.width(), size.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(rect, icon)
        painter.restore()


def icon_thumbnail(kind: str, assets: dict[str, IconAsset] | None = None, size: int = 48) -> QImage:
    image = QImage(size, size, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    icon = resolve_icon(kind, assets)
    if icon is not None:
        painter = QPainter(image)
        try:
            draw_image_icon(painter, icon, QRectF(2, 2, size - 4, size - 4))
        finally:
            painter.end()
    return image
