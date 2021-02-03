from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)


from negmas.java import JavaCallerMixin, to_java
from negmas.outcomes import (
    Issue,
    Outcome,
    outcome_as_dict,
)

from .base import UtilityFunction, UtilityValue

__all__ = [
    "JavaUtilityFunction",
]


class JavaUtilityFunction(UtilityFunction, JavaCallerMixin):
    """A utility function implemented in Java"""

    def __init__(self, java_object, java_class_name: Optional[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_java_bridge(
            java_object=java_object,
            java_class_name=java_class_name,
            auto_load_java=False,
        )
        if java_object is None:
            self._java_object.fromMap(to_java(self))

    def eval(self, offer: "Outcome") -> UtilityValue:
        return self._java_object.call(to_java(outcome_as_dict(offer)))

    def xml(self, issues: List[Issue]) -> str:
        return "Java UFun"
