from v21 import V21EvidencePool, reconcile_evidence


def _pool(*items):
    pool = V21EvidencePool(pool_id="synthetic")
    for pass_id, text, bbox, confidence in items:
        pool.append(pass_id, text, bbox, confidence)
    return pool


def test_same_physical_region_reconciles():
    pool = _pool(
        ("pass_1", "John Smith", [(0, 0), (100, 0), (100, 20), (0, 20)], 0.90),
        ("pass_2", "John Smith", [(2, 1), (98, 1), (98, 19), (2, 19)], 0.95),
    )
    result = reconcile_evidence(pool)
    assert len(result.entities) == 1
    assert result.entities[0].source_token_ids == (
        "synthetic-token-000001",
        "synthetic-token-000002",
    )


def test_identical_text_in_separate_regions_stays_separate():
    pool = _pool(
        ("pass_1", "John Smith", [(0, 0), (50, 0), (50, 20), (0, 20)], 0.90),
        ("pass_2", "John Smith", [(200, 0), (250, 0), (250, 20), (200, 20)], 0.95),
    )
    assert len(reconcile_evidence(pool).entities) == 2


def test_different_tokenization_reconciles_by_geometry():
    pool = _pool(
        ("pass_1", "John", [(0, 0), (40, 0), (40, 20), (0, 20)], 0.90),
        ("pass_1", "Smith", [(45, 0), (95, 0), (95, 20), (45, 20)], 0.90),
        ("pass_2", "John Smith", [(0, 0), (95, 0), (95, 20), (0, 20)], 0.85),
    )
    result = reconcile_evidence(pool)
    assert len(result.entities) == 1
    assert result.entities[0].representative_text == "John Smith"


def test_strong_overlap_can_reconcile_different_text():
    pool = _pool(
        ("pass_1", "J0hn Smith", [(0, 0), (100, 0), (100, 20), (0, 20)], 0.80),
        ("pass_2", "John Smith", [(1, 1), (99, 1), (99, 19), (1, 19)], 0.75),
    )
    assert len(reconcile_evidence(pool).entities) == 1


def test_provenance_and_geometry_are_preserved():
    pool = _pool(
        ("pass_1", "A", [(10, 10), (30, 10), (30, 20), (10, 20)], 0.90),
        ("pass_2", "A longer", [(8, 8), (35, 8), (35, 22), (8, 22)], 0.80),
    )
    result = reconcile_evidence(pool).entities[0]
    assert result.pass_ids == ("pass_1", "pass_2")
    assert result.canonical_bbox == (8.0, 8.0, 35.0, 22.0)


def test_reconciliation_is_deterministic_and_non_destructive():
    pool = _pool(
        ("pass_1", "Alpha", [(0, 0), (40, 0), (40, 20), (0, 20)], 0.90),
        ("pass_2", "Alpha", [(1, 1), (39, 1), (39, 19), (1, 19)], 0.95),
        ("pass_1", "Beta", [(100, 0), (140, 0), (140, 20), (100, 20)], 0.90),
    )
    before = pool.to_dict()
    first = reconcile_evidence(pool)
    second = reconcile_evidence(pool)
    assert first == second
    assert pool.to_dict() == before
