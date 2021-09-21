"""
Defines basic concepts related to outcomes

Outcomes in this package are always assumed to be multi-issue outcomes where  single-issue outcomes can be implemented
as the special case with a single issue.

- Both Continuous and discrete issues are supported. All issue will have names. If none is given, a random name will be
  used. It is HIGHLY recommended to always name your issues.
- Outcomes are dictionaries with issue names as keys and issue values as values.

Examples:

  Different ways to create issues:

  >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')
  ...           , Issue(20, 'count')]
  >>> for _ in issues: print(_)
  price: (0.5, 2.0)
  date: ['2018.10.1', '2018.10.2', '2018.10.3']
  count: (0, 19)

  Outcome example compatible with the given set of issues:

  >>> a = {'price': 1.2, 'date': '2018.10.04', 'count': 4}

"""

from .common import *
from .outcomes import *
from .issues import *


__all__ = common.__all__ + issues.__all__ + outcomes.__all__
