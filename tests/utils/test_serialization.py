# synthcity absolute
from synthcity.utils.serialization import load, save


def test_save_load() -> None:
    obj = {"a": 1, "b": "dssf"}

    objbytes = save(obj)

    reloaded = load(objbytes)

    assert isinstance(reloaded, dict)
    assert reloaded["a"] == 1
    assert reloaded["b"] == "dssf"
