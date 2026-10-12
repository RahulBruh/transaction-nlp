import pytest

from txnlp.clean import clean, features


@pytest.mark.parametrize("raw, expected", [
    ("SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA", "BLUE BOTTLE COFFEE"),
    ("TST* JOES PIZZA #2231 NEW YORK NY", "JOES PIZZA"),
    ("AMZN MKTP US*2K4L9X1", "AMZN MKTP US"),
    ("HILTON #69907 DENVER CO", "HILTON"),
    ("Walgerens 018542 Tempe Az", "WALGERENS"),
    ("SAFEWAY STORE 47124 SAN FRAN CA", "SAFEWAY STORE"),
    ("QT PHOENIX AZ", "QT"),
])
def test_clean(raw, expected):
    assert clean(raw) == expected


def test_clean_never_returns_empty():
    assert clean("12345") != ""


def test_features():
    f = features("SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA")
    assert f["has_processor"] == 1 and f["processor_square"] == 1
    assert f["has_store_number"] == 1 and f["has_location"] == 1
    assert features("NETFLIX.COM")["is_online"] == 1
    assert features("ALDI")["has_processor"] == 0
