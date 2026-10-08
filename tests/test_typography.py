"""Verify shaped glyph families and embedding, rather than requested font names."""

from __future__ import annotations

from importlib.resources import files
from unicodedata import normalize

import pytest
from pypdf import PdfReader
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPainter

from stavellum.engraving.notation import _write_pdf
from stavellum.graphics.font_registry import CJK_FAMILY, LATIN_FAMILY, _font_ids, register_fonts
from stavellum.graphics.qt import ensure_app
from stavellum.graphics.typography import _layout, draw_text, draw_text_rect, text_height


@pytest.mark.parametrize(("text", "family", "style", "italic"), [
    ("English 123!?,()", LATIN_FAMILY, "Roman", False),
    ("cresc. dim. rit. dolce", LATIN_FAMILY, "Italic", True),
    ("乐谱中文，：（）", CJK_FAMILY, "Regular", False),
    ("演奏说明，渐弱", CJK_FAMILY, "Regular", True),
])
def test_actual_glyphs_use_bundled_font_faces(text, family, style, italic):
    layout = _layout(text, 36, italic=italic)
    runs = layout.glyphRuns()
    assert runs
    for run in runs:
        assert run.rawFont().familyName() == family
        assert run.rawFont().styleName() == style
        assert all(run.glyphIndexes())
    assert all(item.format.font().styleStrategy() & QFont.StyleStrategy.NoFontMerging
               for item in layout.formats())


def test_font_resources_are_registered_once_without_recursive_app_creation(monkeypatch):
    ensure_app()
    ids = list(_font_ids)
    assert len(ids) == 3
    assert [QFontDatabase.applicationFontFamilies(item) for item in ids] == [
        [LATIN_FAMILY], [LATIN_FAMILY], [CJK_FAMILY],
    ]

    def forbidden():
        raise AssertionError("register_fonts must not call ensure_app")

    monkeypatch.setattr("stavellum.graphics.qt.ensure_app", forbidden)
    register_fonts()
    assert _font_ids == ids
    directory = files("stavellum").joinpath("fonts")
    assert directory.joinpath("Edwin-Roman.otf").is_file()
    assert directory.joinpath("Edwin-Italic.otf").is_file()
    assert directory.joinpath("SourceHanSerifSC-Regular.otf").is_file()
    assert "SIL OPEN FONT LICENSE" in directory.joinpath("Edwin-LICENSE.txt").read_text()
    assert "SIL OPEN FONT LICENSE" in directory.joinpath("SourceHanSerif-LICENSE.txt").read_text()


def test_utf16_ranges_and_mixed_glyphs_share_baseline():
    layout = _layout("A𠮷中文Z 123", 40)
    assert [(item.start, item.length, item.format.font().family())
            for item in layout.formats()] == [
        (0, 1, LATIN_FAMILY), (1, 4, CJK_FAMILY), (5, 5, LATIN_FAMILY),
    ]
    assert {run.rawFont().familyName() for run in layout.glyphRuns()} == {
        LATIN_FAMILY, CJK_FAMILY,
    }
    baselines = {position.y() for run in layout.glyphRuns() for position in run.positions()}
    assert len(baselines) == 1
    assert all(index for run in layout.glyphRuns() for index in run.glyphIndexes())
    # Shaping an entire Latin range preserves kerning between adjacent letters.
    assert _layout("AV", 40).lineAt(0).naturalTextWidth() < sum(
        _layout(letter, 40).lineAt(0).naturalTextWidth() for letter in "AV"
    )


def test_wrapping_centers_each_line_and_retains_explicit_breaks():
    layout = _layout("Chinese 乐谱 English title words " * 3, 24, width=160,
                     alignment=Qt.AlignmentFlag.AlignCenter, wrap=True)
    assert layout.lineCount() > 1
    for index in range(layout.lineCount()):
        line = layout.lineAt(index)
        assert line.naturalTextWidth() <= 160
        assert line.position().x() + line.naturalTextWidth() / 2 == pytest.approx(80)
    assert _layout("A中文\nTitle 123", 24, width=300, wrap=True).lineCount() == 2


def test_drawing_helpers_paint_mixed_text_and_restore_clip():
    ensure_app()
    image = QImage(400, 160, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.white)
    painter = QPainter(image)
    painter.setPen(Qt.GlobalColor.black)
    draw_text(painter, "ABC乐谱123", 28, QPointF(10, 40))
    draw_text_rect(painter, "中文 English title", 24, QRectF(20, 65, 260, 70), wrap=True)
    assert not painter.hasClipping()
    painter.end()
    assert any(image.pixelColor(x, y).red() < 200
               for x in range(10, 240) for y in range(10, 40))
    assert any(image.pixelColor(x, y).red() < 200
               for x in range(20, 280) for y in range(65, 135))
    assert text_height("ABC乐谱123", 28, 300) > 28


class _SvgPages:
    def getPageCount(self):
        return 2

    def renderToSVG(self, page):
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 700"><path d="M0 0L500 700" stroke="black"/></svg>'


@pytest.mark.integration
def test_pdf_metadata_embeds_both_bundled_fonts_and_wraps_title(tmp_path):
    path = tmp_path / "mixed.pdf"
    title = "English 乐谱中文 title 123 " * 10
    _write_pdf(_SvgPages(), path, title, subtitle="Subtitle 乐曲",
               credits="作曲：Composer  编曲：Arranger")
    reader = PdfReader(path)
    assert len(reader.pages) == 2
    assert reader.metadata.title == title
    for page in reader.pages:
        embedded = {}
        for reference in page["/Resources"]["/Font"].values():
            font = reference.get_object()
            name = str(font["/BaseFont"])
            descendants = font.get("/DescendantFonts", [])
            descriptor = (descendants[0].get_object() if descendants else font)["/FontDescriptor"]
            embedded[name] = any(descriptor.get(key) is not None
                                 for key in ("/FontFile", "/FontFile2", "/FontFile3"))
        assert any("Edwin-Roman" in name and included for name, included in embedded.items())
        assert any("SourceHanSerifSC-Regular" in name and included
                   for name, included in embedded.items())
        # Adobe's CJK font shares glyphs with Kangxi compatibility radicals;
        # Qt's ToUnicode map can therefore extract those equivalent codepoints.
        extracted = "".join(normalize("NFKC", page.extract_text()).split())
        assert "English" in extracted and "乐谱中文" in extracted
    assert "Composer" in reader.pages[0].extract_text()
    assert "Arranger" in reader.pages[0].extract_text()


@pytest.mark.integration
def test_svg_directions_embed_real_italic_font_and_regular_cjk_in_pdf(tmp_path):
    class Pages:
        def getPageCount(self):
            return 1

        def renderToSVG(self, page):
            return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300">
              <g style="font-style: italic; font-family: Arial">
                <text x="30" y="60" font-size="28"><tspan>cresc. 渐强</tspan></text>
              </g>
              <text x="30" y="110" font-size="24">Regular score text</text>
            </svg>'''

    target = tmp_path / "italic-directions.pdf"
    _write_pdf(Pages(), target, "Roman title")
    page = PdfReader(target).pages[0]
    fonts = {}
    for reference in page["/Resources"]["/Font"].values():
        font = reference.get_object()
        name = str(font["/BaseFont"])
        descendants = font.get("/DescendantFonts", [])
        descriptor = (descendants[0].get_object() if descendants else font)["/FontDescriptor"]
        fonts[name] = any(descriptor.get(key) is not None
                          for key in ("/FontFile", "/FontFile2", "/FontFile3"))
    for family in ("Edwin-Italic", "Edwin-Roman", "SourceHanSerifSC-Regular"):
        assert any(family in name and embedded for name, embedded in fonts.items())
    text = normalize("NFKC", page.extract_text())
    assert "cresc." in text and "渐强" in text
