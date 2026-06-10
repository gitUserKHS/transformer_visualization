from backend.scripts import launch_app


def test_launcher_reopens_an_existing_healthy_server(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.delenv("TRANSFORMER_NO_BROWSER", raising=False)
    monkeypatch.setattr(launch_app, "_port_is_open", lambda: True)
    monkeypatch.setattr(launch_app, "_health_is_ready", lambda: True)
    monkeypatch.setattr(launch_app.webbrowser, "open", opened.append)

    assert launch_app.main() == 0
    assert opened == [launch_app.URL]


def test_launcher_rejects_an_unrelated_port_owner(monkeypatch, capsys) -> None:
    monkeypatch.setattr(launch_app, "_port_is_open", lambda: True)
    monkeypatch.setattr(launch_app, "_health_is_ready", lambda: False)

    assert launch_app.main() == 1
    assert "already used by another program" in capsys.readouterr().err


def test_browser_open_can_be_disabled(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setenv("TRANSFORMER_NO_BROWSER", "1")
    monkeypatch.setattr(launch_app.webbrowser, "open", opened.append)

    launch_app._open_browser_when_ready()

    assert opened == []
