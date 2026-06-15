"""Headless-friendly static image export for plotly figures.

``kaleido >= 1.0`` (the maintained line) dropped the Chromium that older
``kaleido < 1.0`` releases bundled, and instead drives a *system* Chrome via
``choreographer``. That is great on a desktop but breaks the common case of
running on a server, container, or CI runner that has no browser installed:
``fig.write_image(...)`` raises ``RuntimeError("Kaleido requires Google Chrome
to be installed")``.

This module wraps ``fig.write_image`` so static export "just works" headless:

* If a browser is already discoverable (kaleido searches ``PATH``, common
  install locations, and the ``BROWSER_PATH`` environment variable), it is used
  and nothing is downloaded.
* Otherwise, unless disabled, a private *Chrome for Testing* build is downloaded
  once into the user cache via :func:`plotly.io.get_chrome` and the export is
  retried. This needs neither ``apt`` nor root.
* Set ``BROWSER_PATH=/path/to/chrome`` to reuse an existing browser and skip the
  download entirely, or provision ahead of time (e.g. in a Docker build) with
  the ``plotly_get_chrome`` CLI command.

Auto-download is on by default and can be turned off by setting
``NEGMAS_AUTO_INSTALL_CHROME`` to ``0`` (or ``false``/``no``); then a missing
browser raises a clear, actionable error instead.

.. note::
    The on-demand download is process-local (memoised per process). When
    running a *parallel* tournament with plotting enabled on a headless
    machine, every worker process would try to download Chrome at once,
    racing on the same cache directory. For that case provision the browser
    once up front -- run ``plotly_get_chrome`` (or set ``BROWSER_PATH`` to an
    existing browser) before launching the run -- so workers just discover it.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["write_image", "ensure_chrome"]

#: Environment variable used to opt out of the on-demand Chrome download.
AUTO_INSTALL_ENV = "NEGMAS_AUTO_INSTALL_CHROME"

# Process-level memo: once an export succeeds (or we have provisioned a
# browser), we know Chrome is available and never re-run the slow detection /
# download path for subsequent figures.
_chrome_ready = False


def _auto_install_enabled() -> bool:
    return os.environ.get(AUTO_INSTALL_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _is_missing_chrome_error(exc: BaseException) -> bool:
    # kaleido >= 1.0 raises a plain RuntimeError whose message mentions Chrome
    # when no usable browser is found.
    return isinstance(exc, RuntimeError) and "chrome" in str(exc).lower()


def _missing_chrome_message(exc: BaseException) -> str:
    return (
        "Saving a plotly figure as a static image needs a Chrome/Chromium "
        "browser, which kaleido>=1.0 does not bundle. None was found and "
        f"automatic installation is disabled ({AUTO_INSTALL_ENV} is off).\n"
        "Fix this in one of these ways:\n"
        "  - run `plotly_get_chrome` once to download a private browser, or\n"
        "  - set BROWSER_PATH=/path/to/chrome to use an existing browser, or\n"
        f"  - unset {AUTO_INSTALL_ENV} to let negmas download one on demand.\n"
        f"Original error: {exc}"
    )


def ensure_chrome() -> None:
    """Download a private Chrome for Testing build if none is available.

    A no-op if a browser was already provisioned this process. Uses
    :func:`plotly.io.get_chrome`, which installs into the user cache without
    needing root or a system package manager.
    """
    global _chrome_ready
    if _chrome_ready:
        return
    import plotly.io as pio

    warnings.warn(
        "No Chrome/Chromium browser found for static image export; downloading "
        "a private Chrome for Testing build (one-time, into the user cache). "
        f"Set BROWSER_PATH to reuse an existing browser, or {AUTO_INSTALL_ENV}=0 "
        "to disable this download.",
        stacklevel=2,
    )
    pio.get_chrome()
    _chrome_ready = True


def write_image(fig: Any, path: str | Path, **kwargs: Any) -> None:
    """``fig.write_image(path)`` that provisions Chrome on demand.

    Behaves exactly like ``fig.write_image`` when a browser is available. On a
    headless machine where kaleido cannot find one, it downloads a private
    browser once (unless :data:`AUTO_INSTALL_ENV` is disabled) and retries. See
    the module docstring for configuration.
    """
    global _chrome_ready
    try:
        fig.write_image(str(path), **kwargs)
        _chrome_ready = True
        return
    except RuntimeError as e:
        if not _is_missing_chrome_error(e):
            raise
        if _chrome_ready or not _auto_install_enabled():
            raise RuntimeError(_missing_chrome_message(e)) from e
    # Browser missing and auto-install allowed: provision once and retry.
    ensure_chrome()
    fig.write_image(str(path), **kwargs)
