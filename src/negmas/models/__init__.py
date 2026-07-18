# -*- coding: utf-8 -*-
"""A package for generalized opponent modeling.

The models here are organized around the four opponent *attributes* that can be learned in
bilateral negotiation, following the taxonomy of Baarslag, Hendrikx, Hindriks & Jonker,
*Learning about the opponent in automated bilateral negotiation: a comprehensive survey of
opponent modeling techniques*, JAAMAS 30:849–898 (2016), Table 2:

- **§5.1 Acceptance strategy** — :mod:`negmas.models.acceptance`
  (:class:`~negmas.models.acceptance.DiscreteAcceptanceModel` and friends).
- **§5.2 Deadline** — :mod:`negmas.models.deadline`
  (:class:`~negmas.models.deadline.DeadlineModel`).
- **§5.3 Preference profile** — :mod:`negmas.models.preferences`
  (:class:`~negmas.models.preferences.OpponentUtilityFunction`) and the ``UFunModel``
  family in :mod:`negmas.sao.components.models` / :mod:`negmas.gb.components.models`.
- **§5.4 Bidding/offering strategy** — :mod:`negmas.models.strategy`
  (:class:`~negmas.models.strategy.OpponentOfferingModel`).
"""

from .acceptance import *  # noqa: F401,F403
from .deadline import *  # noqa: F401,F403
from .preferences import *  # noqa: F401,F403
from .strategy import *  # noqa: F401,F403
from .future import *  # noqa: F401,F403
