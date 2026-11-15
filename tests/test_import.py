import paged_collections


def test_version_is_exposed() -> None:
    assert isinstance(paged_collections.__version__, str)
    assert paged_collections.__version__.count(".") == 2
