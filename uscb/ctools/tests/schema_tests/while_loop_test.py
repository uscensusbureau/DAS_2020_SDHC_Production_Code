"""
Tests while_loop.py
"""
import ast
import pytest
from ctools.schema import while_loop as wl
from ctools.schema import boolean_expression as be
from ctools.schema import boolean_operator as bo
import pandas as pd

@pytest.fixture
def runnable_custom_metadata_while_loop():
    """
    # Create a while loop with metadata such as desc and attrib and an executable syntatically
    # correct while loop.
    # What should hopefully be created via the fixture:
    #
    # while row['num1'] < row['num2']:
    #     print(row['num1'])
    #     print(row['num2'])
    #     print()
    #     row['num1']+=1
    """


    loop_temp = wl.WhileLoop(desc = "While Loop test for loop2", attrib = {"attrib1": "blue", \
        "attrib2": "long"}, condition = be.BooleanExpression(desc = "bool exp for "+
        "loop2", attrib = {"attrib3": "orange", "attrib4": "short"},first_element="num1", \
        second_element="num2", operator= bo.BooleanOperator(desc = "bool_operator for loop2",
        attrib = {"attrib5": "black", "attrib6": "medium"}, op_type = "<")), consequent = \
        ["print(row['num1'])", "print(row['num2'])", "print()", "row['num1']+=1"], \
        indent_spaces = 4)

    return loop_temp


def test_default_metadata():
    """ test constructor without metadata but with same while loop as above"""


    loop_temp = wl.WhileLoop(condition = be.BooleanExpression(first_element="num1", \
    second_element="num2", operator= bo.BooleanOperator(op_type = "<")),
        consequent = ["print(row['num1'])", "print(row['num2'])", "print()", "row['num1']+=1"])

    assert loop_temp.desc==""
    assert loop_temp.attrib == {}
    assert loop_temp.condition.desc==""
    assert loop_temp.condition.attrib=={}
    assert loop_temp.condition.first_element=="num1"
    assert loop_temp.condition.second_element=="num2"
    assert loop_temp.condition.operator.desc==""
    assert loop_temp.condition.operator.attrib=={}
    assert loop_temp.condition.operator.op_type=="<"
    assert loop_temp.consequent==["print(row['num1'])", "print(row['num2'])",\
        "print()", "row['num1']+=1"]
    assert loop_temp.indent_spaces==4
    assert str(loop_temp) == """while row['num1'] < row['num2']:
    print(row['num1'])
    print(row['num2'])
    print()
    row['num1']+=1"""


def test_not_runnable(runnable_custom_metadata_while_loop):
    """ test that code with improper consequent or setup is not runnable """

    # should result in name error
    with pytest.raises(NameError):
        exec(str(runnable_custom_metadata_while_loop))


def test_syntax(runnable_custom_metadata_while_loop):
    """
        test that code with improper consequent or setup is still syntatically sound but fails
        to exec
    """


    # if ast parse works, code syntax is sound
    try:
        ast.parse(str(runnable_custom_metadata_while_loop))
    except SyntaxError:
        assert False

    assert True


def test_runnable(runnable_custom_metadata_while_loop):
    """ test if generated loop is executable without errors upon proper setup """


    # create dataframe with a row that the code can be run on
    temporary_df = pd.DataFrame(columns = ["num1", "num2"])

    temporary_df.loc[0]= [0,1]

    temp_exec = exec("row = temporary_df.iloc[0]" + "\n" \
        + str(runnable_custom_metadata_while_loop))

    # should return nothing upon successful execution
    assert not temp_exec


def test_custom_metadata(runnable_custom_metadata_while_loop):
    """ test custom metadata """


    assert runnable_custom_metadata_while_loop.desc=="While Loop test for loop2"
    assert runnable_custom_metadata_while_loop.attrib == {"attrib1": "blue", "attrib2": "long"}
    assert runnable_custom_metadata_while_loop.condition.desc=="bool exp for loop2"
    assert runnable_custom_metadata_while_loop.condition.attrib==\
        {"attrib3": "orange", "attrib4": "short"}
    assert runnable_custom_metadata_while_loop.condition.first_element=="num1"
    assert runnable_custom_metadata_while_loop.condition.second_element=="num2"
    assert runnable_custom_metadata_while_loop.condition.operator.desc=="bool_operator for loop2"
    assert runnable_custom_metadata_while_loop.condition.operator.attrib==\
        {"attrib5": "black", "attrib6": "medium"}
    assert runnable_custom_metadata_while_loop.condition.operator.op_type=="<"
    assert runnable_custom_metadata_while_loop.consequent==\
        ["print(row['num1'])", "print(row['num2'])", "print()", "row['num1']+=1"]
    assert runnable_custom_metadata_while_loop.indent_spaces==4



def test_repr(runnable_custom_metadata_while_loop, capfd):
    """ test repr function with custom metadata """


    print(repr(runnable_custom_metadata_while_loop))

    out, _ = capfd.readouterr()
    assert out == "While Loop(condition: Boolean Expression(first element:" + \
       " num1, operator: Boolean Operator(type: <), second element: num2)," + \
       " consequent: [\'\"print(row[\\\'num1\\\'])\"\', \'\"print(row[\\\'num2" + \
       "\\\'])\"\', \"\'print()\'\", \'\"row[\\\'num1\\\']+=1\"\'])\n"


def test_str(runnable_custom_metadata_while_loop, capfd):
    """ test str with custom metadata """


    print(str(runnable_custom_metadata_while_loop))

    out, _ = capfd.readouterr()
    assert out == """while row['num1'] < row['num2']:
    print(row['num1'])
    print(row['num2'])
    print()
    row['num1']+=1\n"""


def test_json(runnable_custom_metadata_while_loop, capfd):
    """ test json with custom metadata """


    print(runnable_custom_metadata_while_loop.json_dict())

    out, _ = capfd.readouterr()
    assert out == ("{'desc': 'While Loop test for loop2', 'attrib': "+\
        "{'attrib1': 'blue', 'attrib2': 'long'}, 'condition': {'desc': "+\
        "'bool exp for loop2', 'attrib': {'attrib3': 'orange', 'attrib4': "+\
        "'short'}, 'first_element': 'num1', 'operator': {'desc': 'bool_operator for loop2', "+\
        "'attrib': {'attrib5': 'black', 'attrib6': 'medium'}, 'type': '<'}, "+\
        "'second_element': 'num2'}, 'consequent': [\"print(row['num1'])\", "+\
        "\"print(row['num2'])\", 'print()', \"row['num1']+=1\"]}\n")


def test_dump(runnable_custom_metadata_while_loop, capfd):
    """ test dump with custom metadata """


    print(runnable_custom_metadata_while_loop.dump())

    out, _ = capfd.readouterr()
    assert out == """while row['num1'] < row['num2']:
    print(row['num1'])
    print(row['num2'])
    print()
    row['num1']+=1\nNone\n"""


def test_main(capfd):
    """ test main """


    wl.main()
    out, _ = capfd.readouterr()
    assert out =="while row['True']:\n    pass\n"


def test_defaults(capfd):
    """ test everything being default """


    loop = wl.WhileLoop()
    assert loop.desc == ""
    assert loop.attrib == {}
    assert loop.condition.desc==""
    assert loop.condition.attrib=={}
    assert loop.condition.first_element == "True"
    assert not loop.condition.second_element
    assert loop.condition.operator.desc==""
    assert loop.condition.operator.attrib=={}
    assert not loop.condition.operator.op_type
    assert loop.consequent==['pass']
    assert str(loop) == "while row['True']:\n    pass"
    assert repr(loop) == "While Loop(condition: Boolean Expression(first element: True,"+\
        " operator: Boolean Operator(type: None), second element: None), consequent: [\"'pass'\"])"
    assert loop.json_dict() == {'desc': '', 'attrib': {}, 'condition': {'desc': '', 'attrib':\
        {}, 'first_element': 'True', 'operator': {'desc': '', 'attrib': {}, 'type': None}, \
        'second_element': None}, 'consequent': ['pass']}

    # dump returns none so have to print it. default func of dump is print
    print(loop.dump())
    out, _ = capfd.readouterr()
    assert out == "while row['True']:\n    pass\nNone\n"


def test_assertions_and_errors():
    """
        test that appropriate assertion errors and value errors are raised in cases of
        improper input
    """


    # first element not None or string
    with pytest.raises(AssertionError):
        wl.WhileLoop(condition =be.BooleanExpression(first_element=[1,2]))

    # Second element not None or string
    with pytest.raises(AssertionError):
        wl.WhileLoop(condition =be.BooleanExpression(first_element="a", second_element=[1]))

    # None for first element yet not None for second element
    with pytest.raises(ValueError):
        wl.WhileLoop(condition =be.BooleanExpression(second_element="b"))

    # operator not instance of Boolean Operator
    with pytest.raises(AssertionError):
        wl.WhileLoop(condition =be.BooleanExpression(first_element="a", \
        second_element="b", operator= "some_string"))

    # operator not one of the admissable values resulting in value error.
    # only logical and Comparison operators are allowed
    with pytest.raises(ValueError):
        wl.WhileLoop(condition =be.BooleanExpression(first_element="a", \
        second_element="b", operator= bo.BooleanOperator(op_type= "is")))
