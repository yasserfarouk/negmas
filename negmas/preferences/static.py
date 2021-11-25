__all__ = ["StaticPreferences", "DynamicPreferences"]


class StaticPreferences:
    """
    A mixin that signals to the user that the preferences/ufun is static.

    A static ufun is one that does guarantees that calls with the same outcome
    return the same value every time and can be cached.
    """

    def is_dymanic(self) -> bool:
        return False


class DynamicPreferences:
    """
    A mixin that signals to the user that the preferances/ufun is dynamic.

    A dynamic ufun is one that does not guarantee that calls with the same outcome return the same value every time.
    """

    def is_dymanic(self) -> bool:
        return True
