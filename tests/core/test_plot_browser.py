"""Tests for headless-friendly plotly static export (negmas.plots._browser)."""

from __future__ import annotations

import pytest

from negmas.plots import _browser


def test_is_missing_chrome_error():
    assert _browser._is_missing_chrome_error(
        RuntimeError("Kaleido requires Google Chrome to be installed")
    )
    assert _browser._is_missing_chrome_error(RuntimeError("no chrome found"))
    # unrelated runtime errors and other exception types must not match
    assert not _browser._is_missing_chrome_error(RuntimeError("disk full"))
    assert not _browser._is_missing_chrome_error(ValueError("chrome"))


@pytest.mark.parametrize(
    "value,enabled",
    [
        (None, True),  # default: on
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_auto_install_enabled(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(_browser.AUTO_INSTALL_ENV, raising=False)
    else:
        monkeypatch.setenv(_browser.AUTO_INSTALL_ENV, value)
    assert _browser._auto_install_enabled() is enabled


class _FakeFig:
    """A plotly-figure stand-in that fails N times with a missing-chrome error."""

    def __init__(self, fails: int):
        self.fails = fails
        self.calls = 0

    def write_image(self, path, **kwargs):
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("Kaleido requires Google Chrome to be installed")


def test_write_image_succeeds_without_provisioning(monkeypatch):
    # if the first export works, ensure_chrome must never be called
    monkeypatch.setattr(_browser, "_chrome_ready", False)
    called = {"ensure": False}
    monkeypatch.setattr(
        _browser, "ensure_chrome", lambda: called.__setitem__("ensure", True)
    )
    fig = _FakeFig(fails=0)
    _browser.write_image(fig, "/tmp/_negmas_test.png")
    assert fig.calls == 1
    assert called["ensure"] is False


def test_write_image_provisions_then_retries(monkeypatch):
    # first export fails (no chrome), auto-install on -> provision + retry once
    monkeypatch.setattr(_browser, "_chrome_ready", False)
    monkeypatch.delenv(_browser.AUTO_INSTALL_ENV, raising=False)
    provisioned = {"n": 0}
    monkeypatch.setattr(
        _browser, "ensure_chrome", lambda: provisioned.__setitem__("n", 1)
    )
    fig = _FakeFig(fails=1)
    _browser.write_image(fig, "/tmp/_negmas_test.png")
    assert provisioned["n"] == 1
    assert fig.calls == 2  # failed once, retried once


def test_write_image_actionable_error_when_disabled(monkeypatch):
    monkeypatch.setattr(_browser, "_chrome_ready", False)
    monkeypatch.setenv(_browser.AUTO_INSTALL_ENV, "0")
    fig = _FakeFig(fails=1)
    with pytest.raises(RuntimeError) as ei:
        _browser.write_image(fig, "/tmp/_negmas_test.png")
    msg = str(ei.value)
    assert "plotly_get_chrome" in msg and "BROWSER_PATH" in msg
    assert fig.calls == 1  # no retry when disabled


def test_write_image_reraises_unrelated_runtime_error(monkeypatch):
    monkeypatch.setattr(_browser, "_chrome_ready", False)
    monkeypatch.setattr(
        _browser, "ensure_chrome", lambda: pytest.fail("must not provision")
    )

    class _Boom:
        def write_image(self, path, **kwargs):
            raise RuntimeError("some other kaleido failure")

    with pytest.raises(RuntimeError, match="some other kaleido failure"):
        _browser.write_image(_Boom(), "/tmp/_negmas_test.png")
