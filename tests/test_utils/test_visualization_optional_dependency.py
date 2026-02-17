import pytest

import cbfkit.utils.visualization as visualization


def test_require_visualization_raises_clear_error_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(visualization, "HAS_MATPLOTLIB", False)

    with pytest.raises(ImportError, match=r"cbfkit\[vis\]"):
        visualization.require_visualization()
