from __future__ import annotations

import os

import pkg_resources
import pytest

from negmas import AspirationNegotiator, load_genius_domain_from_folder


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/scenarios"
    )


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + "/other/S-1NIKFRT-1"
    load_genius_domain_from_folder(folder_name, n_discretization=10)


def test_simple_run_with_aspiration_agents():
    file_name = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    assert os.path.exists(file_name)
    domain = load_genius_domain_from_folder(file_name).to_single_issue()
    mechanism = domain.make_session(AspirationNegotiator, n_steps=100, time_limit=30)
    assert mechanism is not None
    mechanism.run()
