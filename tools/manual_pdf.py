"""Render `docs/使用说明.md` to `docs/使用说明.pdf`. Run it after editing the manual.

    python tools/manual_pdf.py
    python tools/manual_pdf.py --out somewhere/else.pdf --keep-html

Three choices worth explaining, because none of them is the obvious one:

**No Markdown library.** The manual uses six constructs -- headings, paragraphs, rules,
lists, tables, fenced code -- and nothing else (no images, no links, no blockquotes; the
inventory is asserted by `tests/test_manual_pdf.py`). A converter for exactly that is
shorter than the argument for adding a dependency to a project whose only runtime
dependencies are numpy and Pillow. If the manual ever grows a construct this does not
know, it says so loudly rather than dropping it silently -- see `_inline` and `convert`.

**Chrome/Edge as the PDF writer.** The alternative is reportlab, which means registering
a CJK font, measuring text by hand and laying tables out in code. A browser already does
CJK line-breaking, table column sizing and widow control, and every Windows machine has
Edge. The cost is that it is a subprocess, which is why this checks the output exists and
is a real PDF instead of trusting the exit code.

**Microsoft YaHei, stated explicitly.** The default sans on a Chinese Windows renders
this fine, but the PDF has to look the same when it is opened on a machine that defaults
to something else, and the browser embeds whatever it actually used.
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "使用说明.md"
BROWSERS = (
    pathlib.Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    pathlib.Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    pathlib.Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    pathlib.Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
)

CSS = """
@page { size: A4; margin: 16mm 14mm 14mm; }

:root {
    --ink: #1f2328;
    --muted: #5b6672;
    --line: #d8dee6;
    --wash: #f4f6f9;
    --accent: #2563eb;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    color: var(--ink);
    background: #ffffff;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    /* Chinese text with no spaces justifies cleanly; ragged right leaves holes. */
    text-align: justify;
}

h1 {
    font-size: 22pt;
    margin: 0 0 4pt;
    letter-spacing: 0.5pt;
}

h1 + p {
    color: var(--muted);
    margin-top: 0;
}

h2 {
    font-size: 14pt;
    margin: 20pt 0 8pt;
    padding-bottom: 4pt;
    border-bottom: 2px solid var(--accent);
}

h3 {
    font-size: 11.5pt;
    margin: 14pt 0 6pt;
    color: var(--accent);
}

/* A heading at the foot of a page with its content overleaf is the single ugliest
   thing a printed manual does. */
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }

p { margin: 6pt 0; }

hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 16pt 0;
}

ul, ol { margin: 6pt 0; padding-left: 20pt; }
li { margin: 3pt 0; }

strong { font-weight: 700; }

code {
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei UI", monospace;
    font-size: 0.9em;
    background: var(--wash);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0 3px;
    /* A long path must be allowed to wrap, or it pushes a table column off the page --
       but only as a last resort. `word-break: break-all` split `all` into `al` + `l`
       mid-sentence, which reads as a typo rather than a wrap. */
    word-break: normal;
    overflow-wrap: break-word;
}

pre {
    background: var(--wash);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 8pt 10pt;
    margin: 8pt 0;
    white-space: pre-wrap;
    break-inside: avoid;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 9.5pt;
    word-break: normal;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 9.5pt;
    /* Chinese cells have no spaces to break on, so let the browser size the columns
       from their content rather than dividing the width equally. */
    table-layout: auto;
}

th, td {
    border: 1px solid var(--line);
    padding: 5pt 7pt;
    text-align: left;
    vertical-align: top;
    line-height: 1.6;
}

th {
    background: var(--wash);
    font-weight: 700;
    white-space: nowrap;
}

/* A row split across a page break loses the reader; a whole table that cannot fit
   still has to break, which is why this is on the row and not the table. */
tr { break-inside: avoid; page-break-inside: avoid; }
thead { display: table-header-group; }

/* The first column of these tables is the label; keeping it narrow stops one long
   explanation from squeezing it to one character per line. */
td:first-child { white-space: nowrap; }
td:first-child:empty { white-space: normal; }
"""

FENCE = re.compile(r"^```(\w*)\s*$")
HEADING = re.compile(r"^(#{1,6}) +(.*)$")
RULE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
BULLET = re.compile(r"^ *[-*] +(.*)$")
NUMBER = re.compile(r"^ *\d+\. +(.*)$")
SEPARATOR = re.compile(r"^\|[\s|:-]+\|$")
UNSUPPORTED = re.compile(r"!\[[^]]*]\(|\[[^]]+]\([^)]+\)|^ *> ")


def _inline(text: str) -> str:
    """`code`, **bold**, and nothing else -- everything else is escaped, not interpreted.

    Order matters: code spans are lifted out first so that a `**` inside one survives as
    literal text, which is the whole reason a manual quotes code in the first place.
    """
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(html.escape(match.group(1)))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{spans[int(m.group(1))]}</code>", text)


def _join(left: str, right: str) -> str:
    """Join two source lines of one paragraph or list item the way each script wants.

    Chinese wraps between any two characters, so a source line break there must vanish:
    a space would open a visible gap mid-sentence. Latin words need that space or they
    fuse. So the space appears only when there is a word character on both sides -- the
    one place where the break was carrying meaning.
    """
    if not left or not right:
        return left + right
    joiner = (
        " "
        if (left[-1].isalnum() and left[-1].isascii())
        and (right[0].isalnum() and right[0].isascii())
        else ""
    )
    return left + joiner + right


def _row(line: str, cell: str) -> str:
    """One table row. A leading and trailing `|` are required, as they are in the source."""
    body = line.strip().strip("|")
    cells = "".join(f"<{cell}>{_inline(part.strip())}</{cell}>" for part in body.split("|"))
    return f"<tr>{cells}</tr>"


def convert(markdown: str) -> str:
    """The manual's Markdown subset, as HTML. Refuses what it cannot render faithfully."""
    out: list[str] = []
    lines = markdown.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()

        if (fence := FENCE.match(line)) is not None:
            index += 1
            block: list[str] = []
            while index < len(lines) and not FENCE.match(lines[index].rstrip()):
                block.append(lines[index])
                index += 1
            index += 1
            language = f' class="language-{fence.group(1)}"' if fence.group(1) else ""
            out.append(f"<pre><code{language}>{html.escape(chr(10).join(block))}</code></pre>")
            continue

        if not line:
            index += 1
            continue

        if (heading := HEADING.match(line)) is not None:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if RULE.match(line) is not None:
            out.append("<hr>")
            index += 1
            continue

        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].rstrip())
                index += 1
            head, body = rows[0], rows[1:]
            if body and SEPARATOR.match(body[0]):
                body = body[1:]
            cells = "".join(_row(row, "td") for row in body)
            out.append(f"<table><thead>{_row(head, 'th')}</thead><tbody>{cells}</tbody></table>")
            continue

        if BULLET.match(line) is not None or NUMBER.match(line) is not None:
            ordered = NUMBER.match(line) is not None
            pattern = NUMBER if ordered else BULLET
            items: list[str] = []
            while index < len(lines):
                current = lines[index].rstrip()
                if (item := pattern.match(current)) is not None:
                    items.append(item.group(1))
                elif current.startswith(("  ", "\t")) and items:
                    # A continuation line: part of the item above, not a new one.
                    items[-1] = _join(items[-1], current.strip())
                else:
                    break
                index += 1
            tag = "ol" if ordered else "ul"
            entries = "".join(f"<li>{_inline(item)}</li>" for item in items)
            out.append(f"<{tag}>{entries}</{tag}>")
            continue

        paragraph = line
        index += 1
        while index < len(lines):
            following = lines[index].rstrip()
            if not following or following.startswith(("|", "#", "```", "- ", "> ")):
                break
            if RULE.match(following) or NUMBER.match(following):
                break
            paragraph = _join(paragraph, following.strip())
            index += 1
        out.append(f"<p>{_inline(paragraph)}</p>")

    return "\n".join(out)


def check_supported(markdown: str) -> None:
    """Fail on a construct this converter would drop, rather than shipping it missing."""
    offenders = [
        f"  line {number}: {line.strip()}"
        for number, line in enumerate(markdown.split("\n"), 1)
        if UNSUPPORTED.search(line)
    ]
    if offenders:
        raise SystemExit(
            "the manual now uses Markdown this converter does not handle (images, links "
            "or blockquotes). Teach `convert` about it -- do not let it disappear from "
            "the PDF:\n" + "\n".join(offenders)
        )


def page(title: str, body: str) -> str:
    return (
        f'<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def find_browser() -> pathlib.Path:
    for candidate in BROWSERS:
        if candidate.is_file():
            return candidate
    for name in ("msedge", "chrome"):
        if (found := shutil.which(name)) is not None:
            return pathlib.Path(found)
    raise SystemExit(
        "no Edge or Chrome found, and one of them is what prints the PDF. Install "
        f"either, or add its path to BROWSERS in {pathlib.Path(__file__).name}."
    )


def print_pdf(browser: pathlib.Path, source: pathlib.Path, target: pathlib.Path) -> None:
    """Drive the browser's headless print. The result is checked, not assumed.

    A profile directory is passed because headless refuses to share the one a running
    browser already holds -- without it this fails whenever the user has Edge open,
    which is most of the time.
    """
    with tempfile.TemporaryDirectory(prefix="vrc-pdf-") as profile:
        result = subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={target}",
                source.resolve().as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    if not target.is_file() or target.stat().st_size == 0:
        raise SystemExit(
            f"{browser.name} produced no PDF (exit {result.returncode}).\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if target.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"{target} is not a PDF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=MANUAL, help="the .md to render")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="default: alongside it")
    parser.add_argument("--keep-html", action="store_true", help="keep the intermediate HTML")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        raise SystemExit(f"no manual at {args.source}")
    target = args.out or args.source.with_suffix(".pdf")
    target.parent.mkdir(parents=True, exist_ok=True)

    markdown = args.source.read_text(encoding="utf-8")
    check_supported(markdown)
    document = page(args.source.stem, convert(markdown))

    intermediate = target.with_suffix(".html")
    keep = args.keep_html
    intermediate.write_text(document, encoding="utf-8")
    try:
        browser = find_browser()
        print(f"$ {browser.name} --headless --print-to-pdf")
        print_pdf(browser, intermediate, target)
    finally:
        if not keep:
            intermediate.unlink(missing_ok=True)

    print(f"{target}  {target.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
