from os import walk
import os

import pkg_resources
import pytest

from negmas import load_genius_domain_from_folder, AspirationNegotiator


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename('negmas', resource_name='tests/data/scenarios')


def test_importing_all_without_exceptions(capsys, scenarios_folder):
    with capsys.disabled():
        base = scenarios_folder
        nxt = 1
        for root, dirs, files in walk(base):
            if len(files) == 0 or len(dirs) != 0:
                continue
            # print(f'{nxt:05}: Importing {root}', flush=True)
            load_genius_domain_from_folder(root)
            nxt += 1


def test_importing_all_single_issue_without_exceptions(capsys, scenarios_folder):
    with capsys.disabled():
        base = scenarios_folder
        nxt, success = 0, 0
        for root, dirs, files in walk(base):
            if len(files) == 0 or len(dirs) != 0:
                continue
            try:
                domain, _, _ = load_genius_domain_from_folder(root, force_single_issue=True
                                                              , max_n_outcomes=10000)
            except Exception as x:
                print(f'Failed on {root}')
                raise x
            nxt += 1
            success += domain is not None
            # print(f'{success:05}/{nxt:05}: {"Single " if domain is not None else "Multi--"}outcome: {root}', flush=True)
