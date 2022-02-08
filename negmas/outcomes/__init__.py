"""
Defines basic concepts related to outcomes

Outcomes in this package are always assumed to be multi-issue outcomes where  single-issue outcomes can be implemented
as the special case with a single issue.

- Both Continuous and discrete issues are supported. All issue will have names. If none is given, a random name will be
  used. It is HIGHLY recommended to always name your issues.
- Outcomes are dictionaries with issue names as keys and issue values as values.

Examples:

  Different ways to create issues:

  >>> issues = [make_issue((0.5, 2.0), 'price'), make_issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')
  ...           , make_issue(20, 'count')]
  >>> for _ in issues: print(_)
  price: (0.5, 2.0)
  date: ['2018.10.1', '2018.10.2', '2018.10.3']
  count: (0, 19)

  Outcome example compatible with the given set of issues:

  >>> a = {'price': 1.2, 'date': '2018.10.04', 'count': 4}

"""
from __future__ import annotations


from .common import *
from .protocols import *
from .base_issue import *
from .callable_issue import *
from .categorical_issue import *
from .contiguous_issue import *
from .continuous_issue import *
from .ordinal_issue import *
from .range_issue import *
from .cardinal_issue import *
from .infinite import *
from .issue_ops import *
from .outcome_ops import *
from .outcome_space import *

__all__ = (
    common.__all__
    + protocols.__all__
    + base_issue.__all__
    + callable_issue.__all__
    + categorical_issue.__all__
    + ordinal_issue.__all__
    + range_issue.__all__
    + cardinal_issue.__all__
    + contiguous_issue.__all__
    + continuous_issue.__all__
    + infinite.__all__
    + issue_ops.__all__
    + outcome_ops.__all__
    + outcome_space.__all__
)
