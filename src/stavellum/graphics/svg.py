"""Normalize Verovio's nested SVG viewport for Qt's SVG renderer."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .font_registry import CJK_FAMILY, LATIN_FAMILY
from .typography import script_runs

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

_TEXT_VISUAL_ATTRIBUTES = {
    "font-size", "font-family", "font-style", "font-weight", "font-variant",
    "text-decoration", "letter-spacing", "fill", "text-anchor",
}


def _text_attributes(node: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    attrs = {**inherited, **node.attrib}
    for declaration in node.get("style", "").split(";"):
        name, separator, value = declaration.partition(":")
        name = name.strip()
        if separator and name in _TEXT_VISUAL_ATTRIBUTES - {"fill"}:
            attrs[name] = value.strip()
    return attrs


def _flatten_text(node: ET.Element, inherited: dict[str, str]) -> None:
    """Qt only reliably positions plain text and one level of styled tspans.

    Verovio nests tspans inside wrapper tspans and puts the text position on a
    wrapper.  Keep the visual attributes but remove that structural nesting.
    """
    pieces: list[tuple[str, dict[str, str]]] = []

    def collect(element: ET.Element, inherited: dict[str, str]) -> None:
        attrs = _text_attributes(element, inherited)
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in ("text", "tspan") and element.text and element.text.strip():
            pieces.append((element.text, attrs))
        for child in element:
            if child.tag.rsplit("}", 1)[-1] == "tspan":
                collect(child, attrs)
            if child.tail and child.tail.strip():
                pieces.append((child.tail, attrs))

    collect(node, inherited)
    if not pieces:
        return
    first_attrs = pieces[0][1]
    for attribute in ("x", "y", "text-anchor", "font-style", "font-weight", "font-family"):
        if node.get(attribute) is None and attribute in first_attrs:
            node.set(attribute, first_attrs[attribute])
    sizes = [attrs["font-size"] for _, attrs in pieces if attrs.get("font-size", "").rstrip("px") not in ("", "0", "0.0")]
    if node.get("font-size", "").rstrip("px") in ("", "0", "0.0"):
        node.set("font-size", sizes[0] if sizes else "405px")
    # Inline CSS overrides presentation attributes in Qt. Flatten the text's
    # style into explicit attributes along with its nested tspan styles.
    remaining_style = ";".join(
        declaration for declaration in node.get("style", "").split(";")
        if declaration.partition(":")[0].strip() not in _TEXT_VISUAL_ATTRIBUTES
        and declaration.strip()
    )
    if remaining_style:
        node.set("style", remaining_style)
    else:
        node.attrib.pop("style", None)
    # Musical tempo marks are private SMuFL font codepoints.  The score exporter
    # normally supplies a BPM label; tolerate existing SVG fixtures as well.
    has_private_glyph = any(any(0xE000 <= ord(char) <= 0xF8FF for char in text) for text, _ in pieces)
    if has_private_glyph:
        text = "".join(text for text, _ in pieces)
        text = "".join(char for char in text if not 0xE000 <= ord(char) <= 0xF8FF)
        text = text.replace("=", "").strip()
        for child in list(node):
            node.remove(child)
        node.text = "BPM " + text
        normal_size = next((attrs.get("font-size") for text, attrs in pieces if not any(0xE000 <= ord(char) <= 0xF8FF for char in text) and attrs.get("font-size", "").rstrip("px") not in ("", "0", "0.0")), "405px")
        node.set("font-size", normal_size)
        node.set("font-family", LATIN_FAMILY)
        node.set("font-style", "normal")
        return
    for child in list(node):
        node.remove(child)
    node.text = None
    node.set("font-family", LATIN_FAMILY)
    for text, attrs in pieces:
        for text_run, cjk in script_runs(text):
            child = ET.SubElement(node, f"{{{SVG_NS}}}tspan")
            child.text = text_run
            for attribute in _TEXT_VISUAL_ATTRIBUTES - {"text-anchor"}:
                if attribute in attrs:
                    child.set(attribute, attrs[attribute])
            child.set("font-family", CJK_FAMILY if cjk else LATIN_FAMILY)
            child.set("font-style", "italic" if not cjk and attrs.get("font-style") in ("italic", "oblique") else "normal")


def normalize_svg(svg: str, foreground: str = "white") -> str:
    root = ET.fromstring(svg)
    inner = next((e for e in root if e.tag == f"{{{SVG_NS}}}svg"), None)
    if inner is not None:
        if inner.get("viewBox"):
            root.set("viewBox", inner.get("viewBox"))
        inner.tag = f"{{{SVG_NS}}}g"
        for attr in ("viewBox", "width", "height", "x", "y", "overflow"):
            inner.attrib.pop(attr, None)
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)
    root.set("color", foreground)
    root.set("fill", foreground)
    # Qt does not consistently apply Verovio's stylesheet to nested use/paths.
    # Musical glyphs remain vector paths; transparent bounding boxes stay invisible.
    for parent in root.iter():
        for child in list(parent):
            if "bounding-box" in child.get("class", "").split():
                parent.remove(child)
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if node.get("color") is not None:
            node.set("color", foreground)
        if tag in ("path", "polygon", "polyline", "ellipse", "rect", "line"):
            node.set("stroke", foreground)
        if node.get("fill") not in ("none", "transparent") and tag not in ("svg", "style"):
            node.set("fill", foreground)
        if tag in ("text", "tspan"):
            node.set("fill", foreground)
    def flatten_texts(node: ET.Element, inherited: dict[str, str]) -> None:
        attrs = _text_attributes(node, inherited)
        if node.tag.rsplit("}", 1)[-1] == "text":
            _flatten_text(node, inherited)
            return
        inherited = {key: value for key, value in attrs.items() if key in _TEXT_VISUAL_ATTRIBUTES}
        for child in node:
            flatten_texts(child, inherited)

    flatten_texts(root, {})
    return ET.tostring(root, encoding="unicode")
