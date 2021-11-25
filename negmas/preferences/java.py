from typing import TYPE_CHECKING, List, Optional

from negmas.java import JavaCallerMixin, to_java
from negmas.outcomes import Issue, Outcome, outcome2dict

from .base import UtilityValue
from .base_crisp import UtilityFunction
from .static import StaticPreferences

__all__ = [
    "JavaUtilityFunction",
]


class JavaUtilityFunction(StaticPreferences, UtilityFunction, JavaCallerMixin):
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
        return self._java_object.call(to_java(outcome2dict(offer)))

    def xml(self, issues: List[Issue]) -> str:
        return "Java UFun"
