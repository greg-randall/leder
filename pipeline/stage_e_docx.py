"""Stage E: Generate a .docx file with Word comments from verified claims.

Each footnote marker in the sourced article becomes a Word comment
anchored to the surrounding text. Upload to Google Drive → open with
Google Docs → all comments appear in the sidebar for collaboration.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from io import BytesIO
from copy import deepcopy

from lxml import etree

# OOXML namespaces
NSMAP = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def _q(namespace: str, tag: str) -> str:
    return f"{{{NSMAP[namespace]}}}{tag}"


def _mk_elem(ns: str, tag: str, attrib: dict = None, text: str = None):
    el = etree.Element(_q(ns, tag), attrib=attrib or {})
    if text:
        el.text = text
    return el


def _mk_run(text: str, bold: bool = False, superscript: bool = False) -> "etree.Element":
    """Create a w:r element with optional formatting."""
    r = etree.Element(_q("w", "r"))
    rPr = etree.SubElement(r, _q("w", "rPr"))
    if bold:
        etree.SubElement(rPr, _q("w", "b"))
    if superscript:
        etree.SubElement(rPr, _q("w", "vertAlign"), {"w:val": "superscript"})
    t = etree.SubElement(r, _q("w", "t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def convert(article_sourced_md: str, claims_json: str, output_docx: str) -> None:
    """Convert sourced article + verified claims to .docx with comments."""

    # Load claims
    with open(claims_json, encoding="utf-8") as f:
        claims_data = json.load(f)
    claims_by_id = {}
    for c in claims_data["claims"]:
        claims_by_id[c["claim_id"]] = c

    # Load article markdown
    with open(article_sourced_md, encoding="utf-8") as f:
        md_text = f.read()

    # Split off the sources section
    body = md_text.split("\n---\n\n## Sources\n")[0]

    # Remove unplaced block
    body = re.sub(r'^# ⚠️ UNPLACED CLAIMS.*?\n\n', '', body, flags=re.DOTALL)

    # Build paragraphs, tracking footnote markers
    paragraphs = []
    for para_text in body.split('\n\n'):
        para_text = para_text.strip()
        if not para_text:
            continue
        if para_text.startswith('# '):
            paragraphs.append(("heading", 1, para_text[2:].strip()))
        elif para_text.startswith('## '):
            paragraphs.append(("heading", 2, para_text[3:].strip()))
        elif para_text.startswith('### '):
            paragraphs.append(("heading", 3, para_text[4:].strip()))
        else:
            paragraphs.append(("body", 0, para_text))

    # Build .docx from template
    template_path = _find_template()
    docx_bytes = _build_docx(template_path, paragraphs, claims_by_id)

    with open(output_docx, "wb") as f:
        f.write(docx_bytes)

    print(f"DOCX written → {output_docx}")
    print(f"Upload to Google Drive, open with Google Docs — comments appear in sidebar")


def _find_template() -> str:
    """Create a minimal .docx template in a temp file."""
    # Start from scratch with a minimal OOXML package
    return _create_minimal_docx()


def _create_minimal_docx() -> str:
    """Create a minimal empty .docx file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    tmp.close()

    # Build the zip with minimal OOXML structure
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:body></w:body></w:document>'
    )

    # Content Types
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/comments.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>'
        '</Types>'
    )

    # .rels
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="word/document.xml"/>'
        '</Relationships>'
    )

    # word/_rels/document.xml.rels
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdComments" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"'
        ' Target="comments.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        # Empty comments file
        comments_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '</w:comments>'
        )
        zf.writestr("word/comments.xml", comments_xml)

    return tmp.name


def _build_docx(template_path: str, paragraphs: list, claims_by_id: dict) -> bytes:
    """Build .docx with comments by directly manipulating the OOXML zip."""

    # Read the template zip
    with open(template_path, "rb") as f:
        template_bytes = f.read()

    # Work in memory
    with zipfile.ZipFile(BytesIO(template_bytes), "r") as zf_in:
        doc_xml = zf_in.read("word/document.xml")
        comments_xml = zf_in.read("word/comments.xml")

    doc_tree = etree.fromstring(doc_xml)
    body = doc_tree.find(_q("w", "body"))

    comments_tree = etree.fromstring(comments_xml)

    comment_id = 0
    comments_el = comments_tree  # Root is w:comments

    # Track footnote refs to assign comment IDs
    fn_pattern = re.compile(r'\[\^(\d+)\]')

    for ptype, level, text in paragraphs:
        if ptype == "heading":
            # Add heading paragraph
            p = etree.SubElement(body, _q("w", "p"))
            pPr = etree.SubElement(p, _q("w", "pPr"))
            pStyle = etree.SubElement(pPr, _q("w", "pStyle"), {_q("w", "val"): f"Heading{level}"})
            r = _mk_run(text)
            p.append(r)
        else:
            # Body paragraph — process inline formatting and footnote markers
            p = etree.SubElement(body, _q("w", "p"))

            # Split text on footnote markers
            parts = fn_pattern.split(text)
            # parts alternates: text, fn_number, text, fn_number, ...

            # If no footnotes, just add a simple paragraph
            if len(parts) == 1:
                # Strip markdown bold/italic
                clean = _strip_markdown_inline(parts[0])
                for run_text in _split_runs(clean):
                    r = _mk_run(run_text)
                    p.append(r)
                continue

            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    clean = _strip_markdown_inline(part)
                    for run_text in _split_runs(clean):
                        r = _mk_run(run_text)
                        p.append(r)
                else:
                    # Footnote number — add comment reference instead of visible marker
                    fn_id = f"c{int(part):04d}"
                    claim = claims_by_id.get(fn_id)

                    # Add comment reference mark
                    cid = comment_id
                    comment_id += 1

                    # Comment range start
                    crs = etree.Element(_q("w", "commentRangeStart"), {_q("w", "id"): str(cid)})
                    p.append(crs)

                    # Comment range end
                    cre = etree.Element(_q("w", "commentRangeEnd"), {_q("w", "id"): str(cid)})
                    p.append(cre)

                    # Comment reference (superscript number)
                    ref_r = etree.Element(_q("w", "r"))
                    ref_rPr = etree.SubElement(ref_r, _q("w", "rPr"))
                    etree.SubElement(ref_rPr, _q("w", "rStyle"), {_q("w", "val"): "CommentReference"})
                    ref_annot = etree.SubElement(ref_r, _q("w", "commentReference"), {_q("w", "id"): str(cid)})
                    p.append(ref_r)

                    # Build comment text
                    if claim:
                        verdict = claim.get("verdict", "?")
                        if verdict == "supported":
                            v_symbol = "✓"
                        elif verdict == "contradicted":
                            v_symbol = "✗"
                        else:
                            v_symbol = "?"

                        rationale = claim.get("rationale", "")
                        source_path = claim.get("source_path", "")
                        source_url = claim.get("source_url", "")

                        comment_lines = [
                            f"{v_symbol} {verdict.upper()} — {claim.get('claim_text', '')}",
                            "",
                            rationale,
                        ]
                        if source_path and source_path != "null":
                            comment_lines.append(f"Source: {source_path}")
                        if source_url and source_url != "null":
                            comment_lines.append(f"URL: {source_url}")
                    else:
                        comment_lines = ["[No verification data]"]

                    # Add comment to comments.xml
                    cmt = etree.SubElement(
                        comments_el,
                        _q("w", "comment"),
                        {
                            _q("w", "id"): str(cid),
                            _q("w", "author"): "Fact Check",
                            _q("w", "date"): datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        },
                    )
                    for cline in comment_lines:
                        cp = etree.SubElement(cmt, _q("w", "p"))
                        cr = etree.SubElement(cp, _q("w", "r"))
                        ct = etree.SubElement(cr, _q("w", "t"))
                        ct.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                        ct.text = cline

    # Serialize back
    doc_str = etree.tostring(doc_tree, xml_declaration=True, encoding="UTF-8", standalone="yes")
    comments_str = etree.tostring(comments_tree, xml_declaration=True, encoding="UTF-8", standalone="yes")

    # Rebuild zip
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with open(template_path, "rb") as tf:
            with zipfile.ZipFile(BytesIO(tf.read()), "r") as zf_in:
                for item in zf_in.infolist():
                    if item.filename == "word/document.xml":
                        zf.writestr(item, doc_str)
                    elif item.filename == "word/comments.xml":
                        zf.writestr(item, comments_str)
                    else:
                        zf.writestr(item, zf_in.read(item.filename))

    return buf.getvalue()


def _strip_markdown_inline(text: str) -> str:
    """Strip basic inline markdown formatting."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text


def _split_runs(text: str) -> list[str]:
    """Split text into runs — for now, return as single run."""
    if not text.strip():
        return []
    return [text]


def run_stage_e(article_sourced_md: str, claims_json: str, output_docx: str = "") -> str:
    if not output_docx:
        output_docx = article_sourced_md.replace(".md", ".docx")
    convert(article_sourced_md, claims_json, output_docx)
    return output_docx
