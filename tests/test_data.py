from txnlp.data import CATEGORIES, MERCHANTS, generate, split_merchants


def test_generation_is_deterministic():
    assert generate(MERCHANTS, 3, seed=42) == generate(MERCHANTS, 3, seed=42)


def test_held_out_merchants_never_in_training():
    seen, held = split_merchants()
    assert not {m.name for m in seen} & {m.name for m in held}
    train_merchants = {r["merchant"] for r in generate(seen, 5, seed=1)}
    assert not train_merchants & {m.name for m in held}


def test_every_category_has_seen_and_held_out_merchants():
    seen, held = split_merchants()
    for cat in CATEGORIES:
        assert any(m.category == cat for m in seen)
        assert any(m.category == cat for m in held)
