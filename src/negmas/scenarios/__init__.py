from pathlib import Path
from negmas.inout import Scenario


def load_scenario(name: str):
    """Loads a scenario from the negmas distribution"""
    return Scenario.load(Path(__file__).parent/name)
