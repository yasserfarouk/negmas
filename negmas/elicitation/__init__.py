# -*- coding: utf-8 -*-
""""
The interface to all negotiators capable of eliciting user preferences before
, and during negotiations.
"""
from __future__ import annotations

from .base import *
from .baseline import *
from .common import *
from .mechanism import *
from .pandora import *
from .queries import *
from .strategy import *
from .user import *
from .voi import *

__all__ = (
    common.__all__
    + strategy.__all__
    + user.__all__
    + queries.__all__
    + base.__all__
    + baseline.__all__
    + pandora.__all__
    + voi.__all__
    + mechanism.__all__
)

import numpy as np

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors
