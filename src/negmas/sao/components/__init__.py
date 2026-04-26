"""
Implements components used to consruct negotiators for SAO.
"""

from .base import *
from .acceptance import *
from .offering import *
from .concession import *
from .selectors import *
from .inverter import *
from .models import *

# Import Genius BOA components from gb
from negmas.gb.components.genius import (
    GeniusAcceptancePolicy,
    GeniusOfferingPolicy,
    GeniusOpponentModel,
    # Acceptance policies
    GACNext,
    GACConst,
    GACTime,
    GACPrevious,
    GACGap,
    GACCombi,
    GACCombiMaxInWindow,
    GACTrue,
    GACFalse,
    GACConstDiscounted,
    GACCombiAvg,
    GACCombiBestAvg,
    GACCombiMax,
    GACCombiV2,
    GACCombiV3,
    GACCombiV4,
    GACCombiBestAvgDiscounted,
    GACCombiMaxInWindowDiscounted,
    GACCombiProb,
    GACCombiProbDiscounted,
    # Offering policies
    GTimeDependentOffering,
    GRandomOffering,
    GBoulwareOffering,
    GConcederOffering,
    GLinearOffering,
    GHardlinerOffering,
    GChoosingAllBids,
    # Opponent models
    GHardHeadedFrequencyModel,
    GDefaultModel,
    GUniformModel,
    GOppositeModel,
    GSmithFrequencyModel,
    GAgentXFrequencyModel,
    GNashFrequencyModel,
    GBayesianModel,
    GScalableBayesianModel,
)

__all__ = (
    base.__all__
    + acceptance.__all__
    + offering.__all__
    + concession.__all__
    + selectors.__all__
    + inverter.__all__
    + models.__all__
    + [
        # Base classes
        "GeniusAcceptancePolicy",
        "GeniusOfferingPolicy",
        "GeniusOpponentModel",
        # Acceptance policies
        "GACNext",
        "GACConst",
        "GACTime",
        "GACPrevious",
        "GACGap",
        "GACCombi",
        "GACCombiMaxInWindow",
        "GACTrue",
        "GACFalse",
        "GACConstDiscounted",
        "GACCombiAvg",
        "GACCombiBestAvg",
        "GACCombiMax",
        "GACCombiV2",
        "GACCombiV3",
        "GACCombiV4",
        "GACCombiBestAvgDiscounted",
        "GACCombiMaxInWindowDiscounted",
        "GACCombiProb",
        "GACCombiProbDiscounted",
        # Offering policies
        "GTimeDependentOffering",
        "GRandomOffering",
        "GBoulwareOffering",
        "GConcederOffering",
        "GLinearOffering",
        "GHardlinerOffering",
        "GChoosingAllBids",
        # Opponent models
        "GHardHeadedFrequencyModel",
        "GDefaultModel",
        "GUniformModel",
        "GOppositeModel",
        "GSmithFrequencyModel",
        "GAgentXFrequencyModel",
        "GNashFrequencyModel",
        "GBayesianModel",
        "GScalableBayesianModel",
    ]
)
