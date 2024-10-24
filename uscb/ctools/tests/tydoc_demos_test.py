import ctools.tydoc as ty
import pytest
import sys

from xml.etree import ElementTree as ET

def test_helpers():
    assert isinstance(ty.p('test1'), ET.Element)
    assert isinstance(ty.h1('test2'), ET.Element)
    assert isinstance(ty.h2('test3'), ET.Element)
    assert isinstance(ty.h3('test4'), ET.Element)
    assert isinstance(ty.pre('test5'), ET.Element)
    assert isinstance(ty.b('test6'), ET.Element)
    assert isinstance(ty.a('test7'), ET.Element)
    assert isinstance(ty.a('test8', href='test'), ET.Element)
    assert isinstance(ty.i('test9'), ET.Element)
    assert isinstance(ty.td('test10'), ET.Element)


def test_demos():
    assert isinstance(ty.demo1(), ty.tydoc)
    assert isinstance(ty.demo2(), ty.tydoc)
    assert isinstance(ty.demo3(), ty.tydoc)
    assert isinstance(ty.demo4(), ty.tydoc)
    assert isinstance(ty.tabledemo1(), ty.tydoc)
    assert isinstance(ty.demo_toc(), ty.tydoc)
    assert ty.jupyter_display_table({'test_val1': (1, 2)}) == None
    assert ty.jupyter_display_table(['test_val1']) == None
    with pytest.raises(ValueError):
        ty.jupyter_display_table('breaker')

def test_tydoc_main(tmpdir):
    test_path = f'{tmpdir}/test_main.txt'
    with open(test_path, 'w') as f:
        sys_save = sys.stdout
        sys.stdout = f
        ty.main(True)
        ty.main(False)
        sys.stdout = sys_save
    with open(test_path, 'r') as f:
        file = f.read()
        assert '---DOM---' in file
        assert '---HTML---' in file
        assert '---LATEX---' in file
        assert '---MD---' in file
