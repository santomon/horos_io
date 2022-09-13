import pytest
from click.testing import CliRunner

from horos_io import cli

@pytest.fixture()
def runner():
    return CliRunner()

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    del response


def test_command_line_interface(runner):
    """Test the CLI."""
    result = runner.invoke(cli.main)
    assert result.exit_code == 0

def test_make_image_info(runner, horos_test_root):
    result = runner.invoke(cli.main, ["image_info", "--root", horos_test_root])
    assert result.exit_code == 0


