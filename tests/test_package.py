import vlmserve


def test_version_is_exposed() -> None:
    assert isinstance(vlmserve.__version__, str)
    assert vlmserve.__version__
