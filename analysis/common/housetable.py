# -*- coding: utf-8 -*-
"""期刊三线表 —— 按本刊参考件的排版习惯

约定（与 supplement material-1111.docx 一致）：
  · 第一列写「缩写, 单位」，不设 Full name 列
  · 全称只在表下缩写表里出现一次
  · 三条线：顶线粗、表头线细、底线粗，其余无框线
"""
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

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


def _run(p, text, size, bold=False, italic=False):
    r = p.add_run(str(text))
    r.font.name = FONT; r.font.size = Pt(size); r.bold = bold; r.italic = italic
    r._element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    return r


def _txt(cell, s, bold=False, size=9, align='center'):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'center': WD_ALIGN_PARAGRAPH.CENTER, 'left': WD_ALIGN_PARAGRAPH.LEFT}[align]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    _run(p, s, size, bold=bold)


def new_doc(landscape=False, margin=1.8):
    d = Document()
    st = d.styles['Normal']
    st.font.name = FONT; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), CJK)
    s = d.sections[0]
    if landscape:
        s.orientation = WD_ORIENT.LANDSCAPE
        s.page_width, s.page_height = s.page_height, s.page_width
    s.left_margin = s.right_margin = Cm(margin)
    return d


def caption(doc, num, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    _run(p, f"{num} ", 10.5, bold=True)
    _run(p, title, 10.5, bold=True)
    return p


def note(doc, text, size=8.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    _run(p, text, size)
    return p


def abbrev(doc, text, size=8.5):
    """缩写表：全称只在这里出现一次"""
    return note(doc, text, size)


def table(doc, header, rows, widths=None, size=8.5, head_size=8.5,
          aligns=None, section_rows=(), first_left=True):
    """header: 一行或两行表头；section_rows: 加粗的分组小标题行索引"""
    if header and not isinstance(header[0], (list, tuple)):
        header = [header]
    nh, ncol = len(header), len(header[0])
    aligns = aligns or (['left'] if first_left else ['center']) + ['center'] * (ncol - 1)
    t = doc.add_table(rows=nh + len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, hrow in enumerate(header):
        for j, h in enumerate(hrow):
            _txt(t.cell(i, j), h, bold=True, size=head_size,
                 align='left' if (j == 0 and first_left) else 'center')
    for i, row in enumerate(rows):
        sec = i in section_rows
        for j, v in enumerate(row):
            _txt(t.cell(nh + i, j), v, bold=sec, size=size,
                 align='left' if (sec or j == 0 and first_left) else aligns[j])
    for j in range(ncol):
        _rule(t.cell(0, j), 'top', 12)
        _rule(t.cell(nh - 1, j), 'bottom', 6)
        _rule(t.cell(nh + len(rows) - 1, j), 'bottom', 12)
    for r in t.rows:
        for c in r.cells:
            for e in ('left', 'right', 'insideH', 'insideV'):
                _rule(c, e, 0)
    for i in range(nh, nh + len(rows) - 1):
        for j in range(ncol):
            _rule(t.cell(i, j), 'bottom', 0)
    for i in range(nh, nh + len(rows)):
        for j in range(ncol):
            _rule(t.cell(i, j), 'top', 0)
    if widths:
        for r in t.rows:
            for j, w in enumerate(widths):
                r.cells[j].width = Cm(w)
    return t
