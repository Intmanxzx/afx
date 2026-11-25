from autofusex.server import server


def test_index_asset_version_changes_when_css_changes(monkeypatch):
    assets = {name: (name.encode(), 'text/plain') for name in server._HASHED_ASSETS}
    monkeypatch.setattr(server, '_static', assets.get)
    html = b'<link href="app.css"><script src="bootstrap.js"></script>'
    first = server._versioned_index(html)
    assets['app.css'] = (b'updated CSS', 'text/css')
    second = server._versioned_index(html)
    assert first != second
    assert second == server._versioned_index(html)
    assert first.split(b'bootstrap.js?v=')[1] == second.split(b'bootstrap.js?v=')[1]
