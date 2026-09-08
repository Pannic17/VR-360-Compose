"""The manual's Markdown -> HTML converter (`tools/manual_pdf.py`).

The PDF itself is printed by Edge or Chrome, which is a subprocess and several seconds,
so it is not exercised here -- `tools/build_exe.py` does that on every release build.
What these tests own is the part that can silently lie: a converter that drops a
construct produces a PDF that looks fine and is missing a paragraph.
"""

from __future__ import annotations

import html as html_mod
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import manual_pdf


def strip_tags(markup: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", "", markup))


def test_the_manual_only_uses_constructs_the_converter_handles() -> None:
    """The guard, run against the real manual -- this is the test that catches an edit.

    Add a link or an image to the manual and this fails, which is the point: the
    alternative is a PDF that quietly lacks it.
    """
    manual_pdf.check_supported(manual_pdf.MANUAL.read_text(encoding="utf-8"))


def test_an_unsupported_construct_is_refused_by_name() -> None:
    with pytest.raises(SystemExit) as caught:
        manual_pdf.check_supported("看 [文档](https://example.com) 吧\n")
    assert "links" in str(caught.value) and "line 1" in str(caught.value)


def test_every_line_of_the_manual_survives_the_conversion() -> None:
    """No paragraph, cell or list item goes missing.

    Compares by non-whitespace characters rather than by line, because the converter
    deliberately joins wrapped source lines back into one paragraph.
    """
    source = manual_pdf.MANUAL.read_text(encoding="utf-8")
    rendered = strip_tags(manual_pdf.convert(source))

    def bare(text: str) -> str:
        # Drop Markdown's own punctuation and all whitespace; what is left is content.
        # The `1.` of an ordered item goes too -- `<ol>` renders the number itself, so
        # it is syntax here in exactly the way `#` and `|` are.
        return re.sub(r"[\s`*|#\-]", "", re.sub(r"^\s*\d+\.\s", "", text))

    missing = [
        line for line in source.split("\n") if bare(line) and bare(line) not in bare(rendered)
    ]
    assert not missing, missing


def test_the_manual_renders_to_a_plausible_document() -> None:
    html = manual_pdf.convert(manual_pdf.MANUAL.read_text(encoding="utf-8"))
    assert html.count("<h1>") == 1
    # 你拿到的是, 来源与输出, 输出模式, 参数, 大概要跑多久, 出问题怎么看
    assert html.count("<table>") == 6, "one per table in the manual"
    assert html.count("<pre>") == 2
    assert "<thead>" in html and "<tbody>" in html
    assert "<img" not in html and "<a " not in html


def test_headings_rules_and_fences() -> None:
    html = manual_pdf.convert("# 标题\n\n---\n\n```bash\nls -la\n```\n")
    assert "<h1>标题</h1>" in html
    assert "<hr>" in html
    assert '<pre><code class="language-bash">ls -la</code></pre>' in html


def test_a_table_keeps_its_header_and_drops_the_separator_row() -> None:
    html = manual_pdf.convert("| 项 | 值 |\n|---|---|\n| 尺寸 | 8K |\n")
    assert "<th>项</th><th>值</th>" in html
    assert "<td>尺寸</td><td>8K</td>" in html
    assert "---" not in html, "the separator row is syntax, not content"


def test_an_empty_leading_header_cell_is_kept() -> None:
    """`| | 说明 |` is how the manual writes a label column with no title."""
    html = manual_pdf.convert("| | 说明 |\n|---|---|\n| 编码 | h264 |\n")
    assert "<th></th><th>说明</th>" in html


def test_a_list_item_continues_onto_the_next_line() -> None:
    html = manual_pdf.convert("- 第一项，说明很长\n  所以换了一行\n- 第二项\n")
    assert html.count("<li>") == 2
    assert "<li>第一项，说明很长所以换了一行</li>" in html


def test_numbered_lists_are_ordered_lists() -> None:
    html = manual_pdf.convert("1. 解压\n2. 双击\n")
    assert html.startswith("<ol>") and "<li>解压</li><li>双击</li>" in html


def test_a_cjk_line_break_leaves_no_gap_but_a_latin_one_keeps_its_space() -> None:
    """The one place a source line break carries meaning is between two Latin words."""
    assert manual_pdf._join("中文", "继续") == "中文继续"
    assert manual_pdf._join("hello", "world") == "hello world"
    assert manual_pdf._join("结尾。", "开头") == "结尾。开头"
    assert manual_pdf._join("8K", "H.264") == "8K H.264"
    # A trailing full stop is not a word character, so no space is invented after it.
    assert manual_pdf._join("行）。", "头显播放") == "行）。头显播放"


def test_code_spans_are_literal_all_the_way_through() -> None:
    html = manual_pdf._inline("用 `**not bold**` 与 `<tag>` 表示")
    assert "<code>**not bold**</code>" in html, "markup inside a code span stays text"
    assert "<code>&lt;tag&gt;</code>" in html
    assert "<strong>" not in html


def test_bold_and_escaping_outside_code() -> None:
    assert manual_pdf._inline("**重要**") == "<strong>重要</strong>"
    assert manual_pdf._inline("a < b & c") == "a &lt; b &amp; c"


def test_the_page_declares_a_cjk_font_and_the_right_language() -> None:
    """A PDF opened on a machine with different defaults still has to look like this."""
    page = manual_pdf.page("使用说明", "<p>x</p>")
    assert 'lang="zh-CN"' in page and 'charset="utf-8"' in page
    assert "Microsoft YaHei" in page
    assert "<title>使用说明</title>" in page
