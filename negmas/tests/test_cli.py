from click.testing import CliRunner

from negmas.scripts.app import cli as main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert len(result.output) > 0
    assert result.exit_code == 0
