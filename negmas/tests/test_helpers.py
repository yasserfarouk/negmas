import sys
from enum import Enum

import pytest

from negmas.helpers import create_loggers, unique_name, is_nonzero_file, ConfigReader
from negmas.helpers import pretty_string


def test_create_loggers_with_default_params(capsys):
    log = create_loggers()

    log.info("Test info")
    _, captured = capsys.readouterr()
    assert captured == ""


def disabled_test_create_loggers_with_file_params(capsys, tmpdir):
    file_name = tmpdir.join("log.txt")

    log = create_loggers(
        file_name=file_name, module_wide_log_file=True, app_wide_log_file=False
    )

    log.info("Test info")
    # _, captured = capsys.readouterr()
    # assert captured == ''

    # with open(file_name, 'r') as f:
    #     flog = f.read()
    #     assert flog.endswith('INFO - Test info\n')

    log.warning("Test message")
    # _, captured = capsys.readouterr()
    # assert captured.endswith('WARNING - Test message\n')

    # with open(file_name, 'r') as f:
    #     flog = f.read()
    #     assert flog.endswith('WARNING - Test message\n')


def test_unique_name_defaults():
    a = unique_name("")
    assert len(a) == 8 + 1 + 6 + 8


def test_unique_name_no_time():
    assert len(unique_name("", add_time=False)) == 8


def test_unique_name_with_path(tmpdir):
    a = unique_name(str(tmpdir))
    assert a.startswith(str(tmpdir))


def test_is_nonzero_file(tmpdir):
    f_name = unique_name("")
    f = tmpdir / f_name
    assert is_nonzero_file(str(f)) is False

    with open(f, "w") as tst_file:
        tst_file.write("")
    assert is_nonzero_file(str(f)) is False

    with open(f, "w") as tst_file:
        tst_file.write("test")
    assert is_nonzero_file(str(f)) is True


def test_pretty_string(capsys):

    assert pretty_string("Test with no components") == "Test with no components"
    assert pretty_string(4) == "4"
    assert pretty_string(2.5) == "2.5"
    assert pretty_string([1, 2, 3]) == "[\n  1\n  2\n  3\n]"
    assert pretty_string((1, 2, 3)) == "[\n  1\n  2\n  3\n]"
    assert pretty_string({"a": 1, "b": 2}) == "{\n  a:1\n  b:2\n}"

    assert pretty_string([1, 2, 3], compact=True) == "[  1  2  3]"
    assert pretty_string((1, 2, 3), compact=True) == "[  1  2  3]"
    assert pretty_string({"a": 1, "b": 2}, compact=True) == '{  "a":1  "b":2}'


def test_config_reader():
    class A(ConfigReader):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test, children = A.from_config(
        config={"a": 10, "b": [1, 2, 3]}, ignore_children=False, scope=locals()
    )
    assert isinstance(test, A)
    assert test.a == 10
    assert test.b == [1, 2, 3]


def test_config_reader_with_enum():
    class E(Enum):
        E1 = 1
        E2 = 2

    class A(ConfigReader):
        def __init__(self, a, b, e: E):
            self.a = a
            self.b = b
            self.e = e

    test, children = A.from_config(
        config={"a": 10, "b": [1, 2, 3], "e:E": 1},
        ignore_children=False,
        scope=locals(),
    )
    assert isinstance(test, A)
    assert test.a == 10
    assert test.b == [1, 2, 3]
    assert test.e == E.E1


def test_config_reader_ignoring_children():
    class A(ConfigReader):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test = A.from_config(config={"a": 10, "b": [1, 2, 3]}, scope=locals())
    assert isinstance(test, A)
    assert test.a == 10
    assert test.b == [1, 2, 3]


class B(ConfigReader):
    def __init__(self, bb, bbdefault=5):
        self.bb = bb
        self.bbdefaults = bbdefault


class Other(ConfigReader):
    def __init__(self, o):
        self.o = o


class C(ConfigReader):
    def __init__(self, cc, b: B = None, others=None):
        self.cc = cc
        self.b = B
        self.others = others

    def set_others(self, others):
        self.others = others


class A(ConfigReader):
    def __init__(self, a, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c


def test_config_reader_with_subobjects():
    test = A.from_config(
        config={
            "a": 10,
            "b": {"bb": 20},
            "c": {
                "cc": 5,
                "b": {"bb": 20},
                "others": [{"o": 1}, {"o": 2}, None, {"o": 3}],
            },
        },
        scope=globals(),
    )

    assert isinstance(test, A)
    assert test.a == 10
    assert isinstance(test.b, B)
    assert isinstance(test.c, C)
    assert isinstance(test.c.others, list)
    assert isinstance(test.c.others[0], Other)
    assert len(test.c.others) == 4
    assert test.c.others[2] is None


if __name__ == "__main__":
    pytest.main(args=[__file__])
