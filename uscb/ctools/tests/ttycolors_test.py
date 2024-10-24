import ctools.ttycolors as ttyc
import io
import sys


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_color():
    # Check one Color
    sys.stdout = io.StringIO()
    args = Namespace(color=['CBLUE'], message='Test', demo=False)
    ttyc.main(args)
    assert sys.stdout.getvalue() == ttyc.CBLUE + 'Test' + ttyc.CEND + '\n'
    # Refresh and Check with multiple Colors
    sys.stdout = io.StringIO()
    args = Namespace(color=['CBLUE', 'CBLINK'], message='Test', demo=False)
    ttyc.main(args)
    expected_out = ttyc.CBLUE + ttyc.CBLINK + 'Test' + ttyc.CEND + '\n'
    assert sys.stdout.getvalue() == expected_out


def test_demo():
    # Run demo, make sure it passes without failure
    # Could do color comparisons but thats what previous test is for
    args = Namespace(color=False, message=False, demo=True)
    ttyc.main(args)
