# Some tests for tytable

import io
import json
import sys
import os
import os.path
import pytest
import tempfile
import warnings

from xml.etree import ElementTree as ET
from os.path import abspath
from os.path import dirname

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

import ctools.tydoc as ty # differentiate as file vs tydoc class
from ctools.tydoc import tydoc,TyTag, tytable, ET, tytable, ATTRIB_ALIGN, ALIGN_CENTER, \
    ALIGN_RIGHT, OPTION_TABLE, OPTION_LONGTABLE, TAG_X_TOC, TAG_BODY, TAG_H1, TAG_A, TAG_TD
from ctools.latex_tools import run_latex,no_latex,LatexException

ATTR_TYPE = 't'
ATTR_VAL = 'v'


class Namespace:
    def __init__(self):
        self.option_enabled = True

    def option(self, option):
        # We are intentionally ignoring the passed in option
        if self.option_enabled:
            return True
        return False

    def enable_option(self):
        self.option_enabled = True

    def disable_option(self):
        self.option_enabled = False

    def savefig(self, buf, format='png'):
        pass


def get_test_cell():
    return TyTag(TAG_TD,attrib={'COLSPAN':2},text='Wide Column')


@pytest.fixture
def test_filepath(tmpfile):
    filepath = f'{tmpfile}/test_file'
    file = open(f'{tmpfile}/test_file', 'w')
    file.write('test file fixture')
    return filepath


@pytest.fixture
def xml_data():
    xml_string = """<?xml version="1.0"?>\n
                <data>\n
                    <state name="Delaware">\n
                        <rank>1</rank>\n
                        <abbreviation>"DE"</abbreviation>\n
                    </state>\n
                    <state name="Pennsylvania">\n
                        <rank>2</rank>\n
                        <abbreviation>"PA"</abbreviation>\n
                    </state>\n
                </data>"""
    return xml_string


@pytest.fixture
def xml_script_data():
    xml_string = """<html>
                        <head><title>test Data</title>
                            <script src="testscript.js"></script>
                        </head>
                        <body><h1></h1></body>
                    </html>"""
    return xml_string


def test_etree_to_dict(xml_data):
    tree = ET.XML(xml_data)
    out = ty.etree_to_dict(tree)
    assert type(out) == dict
    assert 'data' in out.keys()
    assert 'state' in out['data']


def test_is_empty(xml_script_data):
    root = ET.fromstring(xml_script_data)
    # Check if script or link tags contain anything
    assert not ty.is_empty(root.find('.//script'))
    # Check h1 and title tags
    assert ty.is_empty(root.find('.//h1'))
    assert not ty.is_empty(root.find('.//title'))


def test_tytag_option():
    t = TyTag('demo')
    t.set_option("FOO")
    assert t.option("FOO")==True
    assert t.option("BAR")==False
    t.set_option("BAR")
    assert t.option("FOO")==True
    assert t.option("BAR")==True
    t.clear_option("FOO")
    assert t.option("FOO")==False
    assert t.option("BAR")==True


def test_tytable_access():
    """Make sure construction and access methods work properly"""
    t = tytable()
    t.add_head(['x','x-squared','x-cubed'])
    t.add_data([1,1,1])
    t.add_data([2,4,8])
    t.add_data([3,9,27])
    for row in t.rows():
        s = ET.tostring(row,encoding='unicode')
        print(s)
    assert t.get_cell(0,1).text == 'x-squared'
    assert float(t.get_cell(1,1).text) == 1
    assert float(t.get_cell(2,1).text) == 4
    assert float(t.get_cell(3,1).text) == 9


def test_renderer_format():
    # It just returns None idk
    assert ty.Renderer.format() == None


def test_renderer_write():
    # set var for using static renderer classes
    rend = ty.Renderer
    # setup test strings
    beg = 'begin_tag'
    mid = 'middle text'
    end = 'end_tag'
    tail1 = Namespace()
    tail1.tail = 'tail 1\n'
    tail2 = 'tail 2\n'
    with tempfile.TemporaryDirectory() as tmp:
        doc_name = f'{tmp}/test_doc.txt'
        f = open(doc_name, 'w')
        # write to a file using the tydoc functions
        assert rend.write_tag_begin(beg, f)
        assert rend.write_text(None, f, mid)
        f.write('\n')
        assert rend.write_tag_end(end, f)
        assert rend.write_tail(tail1, f)
        assert rend.write_tail(None, f, tail2)
        f.close()
        # after writing using the test functions, check the output
        f = open(doc_name, 'r')
        assert f.readline() == f"\'{beg}\'\n"
        assert f.readline() == mid + '\n'
        assert f.readline() == "--- end repr(doc) ---\n"
        assert f.readline() == tail1.tail
        assert f.readline() == tail2
        f.close()


def test_HTMLRenderer():
    assert ty.HTMLRenderer.format() == ty.FORMAT_HTML


def test_LatexRenderer():
    rend = ty.LatexRenderer
    assert rend.format() == ty.FORMAT_LATEX
    beg = 'begin_tag'
    mid = 'middle_text'
    end = 'end_tag'
    # create namespace object to emulate doc
    doc = Namespace()
    doc.tag = beg
    doc.tail = 'tail_1 \n'
    tail2 = 'tail 2\n'
    # test various write functions
    with tempfile.TemporaryDirectory() as tmp:
        doc_name = f'{tmp}/test_doc.txt'
        f = open(doc_name, 'w')
        assert rend.write_tag_begin(doc, f)
        rend.write_text(doc, f, mid)
        doc.tag = end
        assert rend.write_tag_end(doc, f)
        rend.write_tail(doc, f, None)
        doc.disable_option()
        rend.write_tail(doc, f, None)
        rend.write_tail(doc, f, tail2)
        f.close()
        # check output from write functions
        f = open(doc_name, 'r')
        assert f.readline() == '\n'
        assert f.readline() == '% <BEGIN_TAG>\n'
        assert f.readline() == 'middle\\_{}text\n'
        assert f.readline() == '% </END_TAG>\n'
        assert f.readline() == 'tail\\_{}1 \n'
        assert f.readline() == 'tail_1 \n'
        assert f.readline() == 'tail 2\n'


def test_MarkdownRenderer():
    rend = ty.MarkdownRenderer()
    assert rend.format() == ty.FORMAT_MARKDOWN
    beg = ty.TAG_B
    mid = 'middle_text'
    end = ty.TAG_H1
    doc = Namespace()
    doc.tag = beg
    with tempfile.TemporaryDirectory() as tmp:
        doc_name = f'{tmp}/test_doc.txt'
        f = open(doc_name, 'w')
        assert rend.write_tag_begin(doc, f)
        # Should cause silent KeyError which just passes
        doc.tag = mid
        assert rend.write_tag_begin(doc, f)
        assert rend.write_tag_end(doc, f)
        # Should not cause KeyError
        doc.tag = end
        assert rend.write_tag_end(doc, f)
        f.close()
        # check output from write functions
        f = open(doc_name, 'r')
        assert f.readline() == '**\n'
        assert f.readline() == ''


def test_JsonRenderer():
    assert ty.JsonRenderer.format() == ty.FORMAT_JSON


def test_unsupported_render():
    doc = Namespace()
    with pytest.raises(RuntimeError):
        ty.render(doc, 'string', 'fake_format')


def test_tydoc_safenum():
    assert ty.safenum('5') == 5
    assert ty.safenum('5.0') == 5.0
    assert ty.safenum('5f') == '5f'


def test_tydoc_scalenum():
    assert ty.scalenum('5') == 5
    assert ty.scalenum('5000') == '5K'
    assert ty.scalenum('5000000') == '5M'
    assert ty.scalenum('5000000000') == '5G'
    assert ty.scalenum('5000000000000') == '5T'
    assert ty.scalenum('5.0') == 5.0
    assert ty.scalenum('5f') == '5f'


def test_TyTag_asString():
    wide_cell = TyTag(TAG_TD,attrib={'COLSPAN':2},text='Wide Column')
    assert wide_cell.asString() == '<TD COLSPAN="2">Wide Column</TD>'
    with pytest.raises(ValueError):
        wide_cell.asString(None)


def test_TyTag_save(tmpdir):
    wide_cell = TyTag(TAG_TD,attrib={'COLSPAN':2},text='Wide Column')
    f_name = tmpdir + 'test_file.html'
    # Save text to file
    wide_cell.save(f_name)
    # Grab text to ensure it was saved correctly
    with open(f_name, 'r') as f:
        assert f.read() == '</TD>'
    # Ensure that file is required to have a format
    f_name_fail = tmpdir + 'test_file'
    with pytest.raises(RuntimeError):
        wide_cell.save(f_name_fail)


def test_TyTag_set_attrib():
    wide_cell = TyTag(TAG_TD,attrib={'COLSPAN':2},text='Wide Column')
    d = {ty.ATTRIB_OPTIONS: ty.ALIGN_CENTER}
    assert len(wide_cell.attrib) == 1
    assert wide_cell.attrib['COLSPAN'] == 2
    wide_cell.set_attrib(d)
    # ensure attribs are concatenated
    assert len(wide_cell.attrib) == 2
    assert wide_cell.attrib['COLSPAN'] == 2
    assert wide_cell.attrib[ty.ATTRIB_OPTIONS] == ty.ALIGN_CENTER


def test_TyTag_setText():
    text1 = 'first text'
    text2 = 'second text'
    cell = TyTag(TAG_TD, text=text1)
    assert cell.text == text1
    cell.setText(text2)
    assert cell.text == text2


def test_TyTag_add_tag_elems():
    attrib1 = {'id': 1}
    attrib2 = {'class': 2}
    attrib3 = {'test': 3}
    attrib4 = {'test': 4}
    cell = TyTag(TAG_TD, attrib=attrib1, text='Test Column')

    # check 'id' cannot be in attrib
    with pytest.raises(ValueError):
        cell.add_tag_elems(tag='', attrib=attrib1, id='1')
    cell = TyTag(TAG_TD, attrib=attrib2, text='Test Column')

    # check 'class' cannot be in attrib
    with pytest.raises(ValueError):
        cell.add_tag_elems(tag='', attrib=attrib2, className=3)

    # check tag cannot be <
    with pytest.raises(ValueError):
        cell.add_tag_elems(tag='<', attrib=attrib3, id=4, className=5)

    tytag = cell.add_tag_elems(tag=ty.TAG_TH, attrib=attrib4, id=5, className=6)
    assert type(tytag) == ty.TyTag


def test_TyTag_append_image():
    test_io = io.BytesIO(b'test')
    cell = TyTag(TAG_TD, text='Test Column')
    # check cell can ingest BytesIO object
    cell.append_image(test_io, format='png')
    # check cell can ingest non-BytesIO object
    cell.append_image('fake_file.png', format='png')
    # Checking the output of this function is very difficult, so we don't


def test_TyTag_append_matplotlib():
    fig_replacement = Namespace()
    cell = TyTag(TAG_TD, text='Test Column')
    # check cell can ingest non-BytesIO object
    cell.append_matplotlib(fig_replacement, format='png')
    # Checking the output of this function is very difficult, so we don't


def test_TyTag_simple_functions(tmpdir):
    # test small functions in TyTag like "div" and "a"
    cell = get_test_cell()
    cell.div('test div')
    cell.pre('test pre')
    cell.a('test a')
    cell.b('test b')
    cell.hr()
    cell.json_table()
    cell.ul('test ul')
    cell.li('test li')
    cell.span('test span')
    # once all the additions to the file have been made, render it
    test_path = f'{tmpdir}/test_simple.html'
    with open(test_path, 'w') as file:
        cell.render(file)
    # then read in the file and check it rendered all the functions
    with open(test_path, 'r') as file:
        lines = []
        lines.append('<TD COLSPAN="2">Wide Column<DIV>test div</DIV>\n')
        lines.append('<PRE>test pre</PRE>\n')
        lines.append('<A>test a</A><B>test b</B><HR/>{}<UL>test ul</UL>\n')
        lines.append('<UL>test li</UL>\n')
        lines.append('<SPAN>test span</SPAN>\n')
        lines.append('</TD>')
        for line in lines:
            assert file.readline() == line


@pytest.mark.parametrize("file_format,expected",
    [('html', '<img alt="" src="data:image/html;base64,c29tZSB3b3Jkcw==" />'),
     ('latex', 'test_img_image.png'),
     ('md', 'test_img_image.png')
    ])
def test_EmbeddedImageTag_custom_renderer(file_format, expected, tmpdir):
    test_path = f'{tmpdir}/test_img.{file_format}'
    tag = ty.EmbeddedImageTag(buf=b'some words', format=file_format)
    with open(test_path, 'w') as f:
        tag.custom_renderer(f, format=file_format)
    with open(test_path, 'r') as f:
        assert expected in f.readline()


def test_broken_EmbeddedImageTag(tmpdir):
    fake_format = 'zippy'
    test_path = f'{tmpdir}/test_img.{fake_format}'
    tag = ty.EmbeddedImageTag(buf=b'some words', format=fake_format)
    with open(test_path, 'w') as f:
        with pytest.raises(RuntimeError):
            tag.custom_renderer(f, format=fake_format)


def test_tydoc_write_tag(tmpdir):
    doc = tydoc()
    doc.h1("Table demo")
    test_path = f"{tmpdir}/tydoc_tag_test"
    with open(test_path, 'w') as f:
        out1 = doc.write_tag_begin(f, format=ty.FORMAT_HTML)
        assert out1 == True
        out2 = doc.write_tag_begin(f, format=ty.FORMAT_MARKDOWN)
        assert out2 == False # cant write markdown this way
        out3 = doc.write_tag_end(f, format=ty.FORMAT_HTML)
        assert out3 == True
        out4 = doc.write_tag_end(f, format=ty.FORMAT_MARKDOWN)
        assert out4 == False


def test_tydoc_title():
    doc = tydoc()
    doc.h1("Table demo")
    # the add tag elems stuff that title does is tested elsewhere
    # just make sure this runs
    doc.title('le title')


def test_tydoc_insert_toc():
    doc = tydoc()
    doc.insert_toc()
    doc.insert_toc()


def test_X_TOC(tmpdir):
    toc = ty.X_TOC(attrib={'COLSPAN':2})
    test_path = f"{tmpdir}/X_TOC_tag_test"
    fake_format = 'zippy'
    with open(test_path, 'w') as f:
        out1 = toc.custom_renderer(f, ty.FORMAT_LATEX)
        assert out1 == True
        out2 = toc.custom_renderer(f, fake_format)
        assert out2 == False
        out3 = toc.write_tag_begin(f)
        assert out3 == True
        out4 = toc.write_tag_begin(f, fake_format)
        assert out4 == False
        out5 = toc.write_tag_end(f)
        assert out5 == True
        out6 = toc.write_tag_end(f, fake_format)
        assert out6 == False
    with open(test_path, 'r') as f:
        assert f.readline() == '\\tableofcontents\n'


def test_tag_ignore():
    assert type(ty.tag_ignore()) == type(ty.ET.Element(ty.TAG_TIGNORE))


def test_jsonTable_cells_in_row():
    # Create two objects with tag property, only first should be returned
    t1 = Namespace()
    t1.tag = ty.TAG_TH
    t2 = Namespace()
    t2.tag = 'Fake_tag'
    tr = [t1, t2]
    row_out = ty.jsonTable.cells_in_row(tr)
    assert len(row_out) == 1
    assert row_out[0].tag == t1.tag


def test_jsonTable_custom_renderer(tmpdir):
    ATTR_TYPE = 't'
    j = ty.jsonTable()
    j.add_head(['State','Abbreviation','Population'],
                cell_attribs={ATTRIB_ALIGN:ALIGN_CENTER})
    j.add_data(['Virginia','VA',8001045],
               cell_attribs=[{'id': 'int'},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_RIGHT}])
    j.add_data(['California','CA',37252895],
               cell_attribs=[{'id': 'float'},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_RIGHT}])
    j.add_data(['Pennsylvania', 'PA', 2],
               cell_attribs=[{'id': 'none'},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_RIGHT}])
    j.add_data(['Delaware', 'DE', 1],
               cell_attribs=[{},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_CENTER}])
    with pytest.raises(ValueError):
        j.add_data(['HA', 50],
               cell_attribs=[{'id': 'none'},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_CENTER}])
    j.add_data(['51', 'DE'],
               cell_attribs=[{ATTR_TYPE: 'int'}, {'id': 'none'}])
    j.add_data(['52', 'AZ'],
               cell_attribs=[{ATTR_TYPE: 'float'}, {'id': 'none'}])
    test_path = f'test_json.json'
    with open(test_path, 'w') as f:
        j.custom_renderer(f)
    with open(test_path, 'r') as f:
        data = json.load(f)
        assert data['int'] == 'Virginia'
        assert data['float'] == 'California'
        assert data['none'] == 'AZ'
        assert len(data) == 3


def test_jsonTable_caption():
    j = ty.jsonTable()
    the_caption = 'test caption'
    assert None == j.get_caption()
    j.set_caption(the_caption)
    j.set_fontsize(8)
    assert j.attrib['FONTSIZE'] == '8'
    assert the_caption == j.get_caption()


def test_jsonTable_format_cell():
    j = ty.jsonTable()
    cell1 = ET.Element(ty.TAG_TH)
    dict1 = {'tv': 'test2'}
    j.make_cell('test_tag1', cell1, dict1)
    cell2 = ET.Element('test_tag2')
    j.make_cell('test_tag3', cell2, dict1)

    j2 = ty.jsonTable()
    fake_cell2 = Namespace()
    fake_cell2.attrib = {}
    assert fake_cell2 == j2.format_cell(fake_cell2)
    fake_cell2.typename = ATTR_TYPE
    fake_cell2.attrib = {ATTR_VAL: 'fake_val1'}
    fake_cell2 == j2.format_cell(fake_cell2)
    fake_cell3 = Namespace()
    fake_cell3.attrib = {ATTR_TYPE: None, ATTR_VAL: "3"}
    assert fake_cell3 == j2.format_cell(fake_cell3)
    fake_cell3.attrib[ATTR_TYPE] = 'float'
    assert fake_cell3 == j2.format_cell(fake_cell3)


def test_jsonTable_helpers():
    j = ty.jsonTable()
    j.add_head(['State','Abbreviation','Population'],
                cell_attribs={ATTRIB_ALIGN:ALIGN_CENTER})
    j.add_data(['Virginia','VA',8001045],
               cell_attribs=[{'id': 'int'},{ATTRIB_ALIGN:ALIGN_CENTER},
                              {ATTRIB_ALIGN:ALIGN_RIGHT}])
    rows = j.rows()
    for row in rows:
        assert isinstance(row, ET.Element)
    row0 = j.row(0)
    assert row0 == rows[0]
    assert j.max_cols() == 3
    cell1 = j.get_cell(0, 0)
    assert isinstance(cell1, ET.Element)
    cols = j.col(0)
    assert isinstance(cols, list)


def test_tytable_attribs():
    d2 = tytable()
    d2.set_option(OPTION_LONGTABLE)
    d2.add_head(['State','Abbreviation','Population'],cell_attribs={ATTRIB_ALIGN:ALIGN_CENTER})
    d2.add_data(['Virginia','VA',8001045],
                cell_attribs=[{},{ATTRIB_ALIGN:ALIGN_CENTER},{ATTRIB_ALIGN:ALIGN_RIGHT}])
    d2.add_data(['California','CA',37252895],
                cell_attribs=[{},{ATTRIB_ALIGN:ALIGN_CENTER},{ATTRIB_ALIGN:ALIGN_RIGHT}])
    s = ET.tostring(d2,encoding='unicode')
    assert 'CENTER' in s
    assert d2.get_cell(0,0).attrib[ATTRIB_ALIGN]==ALIGN_CENTER
    assert d2.get_cell(0,1).attrib[ATTRIB_ALIGN]==ALIGN_CENTER
    assert d2.get_cell(0,2).attrib[ATTRIB_ALIGN]==ALIGN_CENTER
    assert ATTRIB_ALIGN not in d2.get_cell(1,0).attrib
    assert d2.get_cell(1,1).attrib[ATTRIB_ALIGN]==ALIGN_CENTER
    assert d2.get_cell(1,2).attrib[ATTRIB_ALIGN]==ALIGN_RIGHT


def test_tydoc_latex(tmpdir):
    """Create a document that tries lots of features and then make a LaTeX document and run LaTeX"""

    doc = tydoc()
    doc.h1("Table demo")

    d2 = doc.table()
    d2.set_option(OPTION_TABLE)
    d2.add_head(['State','Abbreviation','Population'])
    d2.add_data(['Virginia','VA',8001045])
    d2.add_data(['California','CA',37252895])

    d2 = doc.table()
    d2.set_option(OPTION_LONGTABLE)
    d2.add_head(['State','Abbreviation','Population'])
    d2.add_data(['Virginia','VA',8001045])
    d2.add_data(['California','CA',37252895])

    doc.save(os.path.join(tmpdir, "tydoc.tex"), format="latex")

    if no_latex():
        warnings.warn("Cannot run LaTeX tests")
        return
    try:
        run_latex(os.path.join(tmpdir, "tydoc.tex"))
    except LatexException as e:
        warnings.warn("LatexException: "+str(e))


def test_tydoc_toc():
    """Test the Tydoc table of contents feature."""
    doc = tydoc()
    doc.h1("First Head1")
    doc.p("blah blah blah")
    doc.h1("Second Head1 2")
    doc.p("blah blah blah")
    doc.h2("Head 2.1")
    doc.p("blah blah blah")
    doc.h2("Head 2.2")
    doc.p("blah blah blah")
    doc.h3("Head 2.2.1")
    doc.p("blah blah blah")
    doc.h1("Third Head1 3")
    doc.p("blah blah blah")

    # Add a toc
    doc.insert_toc()

    # Make sure that the TOC has a pointer to the first H1
    print(doc.prettyprint())
    key = f".//{TAG_X_TOC}"
    tocs = doc.findall(key)
    assert len(tocs)==1
    toc = tocs[0]

    h1s = doc.findall(".//{}/{}".format(TAG_BODY,TAG_H1))
    assert len(h1s)==3
    h1 = h1s[0]

    # Make sure that they both have the same ID
    id1 = toc.find('.//{}'.format(TAG_A)).attrib['HREF']
    id2 = h1.find('.//{}'.format(TAG_A)).attrib['NAME']

    assert id1==('#'+id2)


def test_tytable_autoid():
    """test the autoid feature"""
    t = tytable()
    t.add_head(['foo','bar','baz'],col_auto_ids=['foo','bar','baz'])
    t.add_data([1,2,3], row_auto_id="row1")
    t.add_data([2,3,4], row_auto_id="row2")
    t.add_data([5,6,7], row_auto_id="row3")
    with tempfile.NamedTemporaryFile(suffix='.autoid.html', mode='w') as tf:
        t.save( tf, format="html")
    # Should read it and do something with it here.


def test_tytable_colspan():
    """test the colspan feature"""
    t = tytable()
    wide_cell = TyTag(TAG_TD,attrib={'COLSPAN':2},text='Wide Column')
    t.add_head(['foo','bar','baz','bif'],col_auto_ids=['foo','bar','baz','bif'])
    t.add_data([1,2,3,4], row_auto_id="row1")
    t.add_data([2,wide_cell,5], row_auto_id="row2")
    t.add_data([3,4,5,6], row_auto_id="row3")

    # Make sure that the colspan is working
    assert t.get_cell(0,0).text == 'foo'
    assert t.get_cell(0,1).text == 'bar'
    assert t.get_cell(0,2).text == 'baz'
    assert t.get_cell(0,3).text == 'bif'

    assert t.get_cell(1,0).text == '1'
    assert t.get_cell(1,1).text == '2'
    assert t.get_cell(1,2).text == '3'
    assert t.get_cell(1,3).text == '4'

    assert t.get_cell(2,0).text == '2'
    assert t.get_cell(2,1).text == 'Wide Column'
    assert t.get_cell(2,2).text == None
    assert t.get_cell(2,3).text == '5'

    with tempfile.NamedTemporaryFile(suffix='.html', mode='w') as tf:
        t.save( tf, format="html")
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w') as tf:
        t.save( tf, format="json")
    # Should read it and do something with it here.
