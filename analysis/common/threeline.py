# -*- coding: utf-8 -*-
"""期刊三线表（top rule / mid rule / bottom rule），Times New Roman，可直投。"""
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

FONT = "Times New Roman"
CJK = "宋体"


def _rule(cell, edge, sz):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders'); tcPr.append(b)
    e = b.find(qn('w:' + edge))
    if e is None:
        e = OxmlElement('w:' + edge); b.append(e)
    e.set(qn('w:val'), 'single' if sz else 'nil')
    e.set(qn('w:sz'), str(sz)); e.set(qn('w:space'), '0'); e.set(qn('w:color'), '000000')


def _txt(cell, s, bold=False, size=9, align='center'):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'center': WD_ALIGN_PARAGRAPH.CENTER, 'left': WD_ALIGN_PARAGRAPH.LEFT,
                   'right': WD_ALIGN_PARAGRAPH.RIGHT}[align]
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(1.5)
    r = p.add_run(str(s))
    r.font.name = FONT; r.font.size = Pt(size); r.bold = bold
    r._element.rPr.rFonts.set(qn('w:eastAsia'), CJK)


def new_doc(landscape=False):
    from docx.enum.section import WD_ORIENT
    d = Document()
    st = d.styles['Normal']
    st.font.name = FONT; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    if landscape:
        s = d.sections[0]
        s.orientation = WD_ORIENT.LANDSCAPE
        s.page_width, s.page_height = s.page_height, s.page_width
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(1.8)
    return d


def add_caption(doc, num, title, note=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{num}  "); r.bold = True; r.font.name = FONT; r.font.size = Pt(10.5)
    r._element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    r = p.add_run(title); r.bold = True; r.font.name = FONT; r.font.size = Pt(10.5)
    r._element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    return p


def add_footnote(doc, text, size=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    r = p.add_run(text); r.font.name = FONT; r.font.size = Pt(size)
    r._element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    return p


def three_line(doc, header, rows, aligns=None, size=9, head_size=9,
               widths=None, n_head_rows=1, section_rows=(), hmerge=(), vmerge=()):
    """header: list[str] 或 list[list[str]]（多行表头）; rows: list[list]
    section_rows: 行索引集合，这些行加粗且左对齐（类别小标题）"""
    if header and not isinstance(header[0], (list, tuple)):
        header = [header]
    ncol = len(header[0])
    aligns = aligns or ['left'] + ['center'] * (ncol - 1)
    t = doc.add_table(rows=len(header) + len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for hi, hrow in enumerate(header):
        for j, h in enumerate(hrow):
            _txt(t.cell(hi, j), h, bold=True, size=head_size,
                 align='left' if j == 0 else 'center')
    for i, row in enumerate(rows):
        sec = i in section_rows
        for j, v in enumerate(row):
            _txt(t.cell(len(header) + i, j), v, bold=sec, size=size,
                 align='left' if sec else aligns[j])
    nh = len(header)
    # 表头分组：横向合并 + 组下短横线；单列表头纵向合并
    for r0, c0, c1 in hmerge:
        txt = t.cell(r0, c0).text
        m = t.cell(r0, c0).merge(t.cell(r0, c1))
        _txt(m, txt, bold=True, size=head_size, align='center')
        for j in range(c0, c1 + 1):
            _rule(t.cell(r0, j), 'bottom', 4)
    for c0 in vmerge:
        txt = t.cell(0, c0).text
        m = t.cell(0, c0).merge(t.cell(nh - 1, c0))
        _txt(m, txt, bold=True, size=head_size,
             align='left' if c0 == 0 else 'center')
    # 三条线
    for j in range(ncol):
        _rule(t.cell(0, j), 'top', 12)                    # top rule
        _rule(t.cell(nh - 1, j), 'bottom', 6)             # mid rule
        _rule(t.cell(len(header) + len(rows) - 1, j), 'bottom', 12)   # bottom rule
    for r in t.rows:                                       # 其余边框全清
        for c in r.cells:
            for e in ('left', 'right', 'insideH', 'insideV'):
                _rule(c, e, 0)
    for i in range(nh, len(header) + len(rows) - 1):
        for j in range(ncol):
            _rule(t.cell(i, j), 'bottom', 0)
    for i in range(nh, len(header) + len(rows)):
        for j in range(ncol):
            _rule(t.cell(i, j), 'top', 0)
    if widths:
        for r in t.rows:
            for j, w in enumerate(widths):
                r.cells[j].width = Cm(w)
    return t
