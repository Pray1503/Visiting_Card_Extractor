"""Provenance regression tests for V21SpatialEvidence.

These tests exist to guarantee one invariant: the pairwise linkage evidence
returned by build_spatial_evidence() must come from the exact same
full-population _region_linkage_graph() call that determines region
membership in build_spatial_representation(). No per-component
recomputation of the adaptive scale may occur.

All geometry below is synthetic and does not encode any real card's
layout, coordinates, text, or dimensions.
"""

from v21 import (
    V21CanonicalEntity,
    V21ReconciliationResult,
    _region_linkage_graph,
    build_spatial_evidence,
    build_spatial_representation,
)


def _entity(number, bbox):
    return V21CanonicalEntity(
        canonical_id=f"canonical-{number:06d}",
        source_token_ids=(f"token-{number:06d}",),
        pass_ids=("pass_1",),
        canonical_bbox=bbox,
        representative_text=f"entity-{number}",
        representative_token_id=f"token-{number:06d}",
    )


def _result(*entities):
    return V21ReconciliationResult(entities=tuple(entities))


def _two_cluster_population():
    """Two well-separated clusters with deliberately different internal
    entity sizes, so a per-component scale recomputation would diverge
    from the full-population scale actually used to build regions."""
    dense_cluster = [
        _entity(1, (0, 0, 10, 10)),
        _entity(2, (0, 20, 10, 30)),
        _entity(3, (0, 40, 10, 50)),
    ]
    sparse_cluster = [
        _entity(4, (1000, 1000, 1100, 1100)),
        _entity(5, (1000, 1300, 1100, 1400)),
        _entity(6, (1000, 1600, 1100, 1700)),
    ]
    return tuple(dense_cluster + sparse_cluster)


def test_evidence_scale_matches_full_population_graph_not_component():
    """The core provenance guarantee: every linked pair's recorded scale
    must equal the scale computed from the FULL entity population, not a
    scale recomputed from only the entities inside its own component."""
    result = _result(*_two_cluster_population())

    _, full_population_evidence = _region_linkage_graph(result.entities)
    expected_scale = full_population_evidence[0].scale

    spatial_evidence = build_spatial_evidence(result)
    assert spatial_evidence.linked_pair_evidence, "expected at least one linked pair"
    for item in spatial_evidence.linked_pair_evidence:
        assert item.scale == expected_scale


def test_evidence_pairs_are_exact_subset_of_full_population_linked_pairs():
    """Every (first, second, scale) triple in the returned evidence must be
    byte-identical to an entry in the single full-population graph result --
    not merely similar, not recomputed."""
    result = _result(*_two_cluster_population())

    _, full_population_evidence = _region_linkage_graph(result.entities)
    full_linked = {
        (item.first_entity_id, item.second_entity_id, item.scale)
        for item in full_population_evidence
        if item.linked
    }

    spatial_evidence = build_spatial_evidence(result)
    returned = {
        (item.first_entity_id, item.second_entity_id, item.scale)
        for item in spatial_evidence.linked_pair_evidence
    }
    assert returned == full_linked


def test_region_membership_matches_evidence_linked_pairs():
    """Two entities appearing together in the evidence as linked must fall in
    the same region, and no evidence pair may ever cross two different
    regions -- this is the structural link between Phase 3's two outputs."""
    result = _result(*_two_cluster_population())
    spatial = build_spatial_representation(result)
    evidence = build_spatial_evidence(result)

    region_of = {}
    for region in spatial.regions:
        for entity_id in region.entity_ids:
            region_of[entity_id] = region.region_id

    for item in evidence.linked_pair_evidence:
        assert region_of[item.first_entity_id] == region_of[item.second_entity_id]


def test_two_distinct_clusters_remain_two_regions():
    """Baseline sanity check for the synthetic population used above --
    confirms the fixture actually exercises the multi-region case the
    provenance bug required to manifest."""
    result = _result(*_two_cluster_population())
    spatial = build_spatial_representation(result)
    assert len(spatial.regions) == 2


def test_row_construction_unaffected_by_evidence_change():
    result = _result(*_two_cluster_population())
    spatial = build_spatial_representation(result)
    row_ids = [row.row_id for region in spatial.regions for row in region.rows]
    # Each dense/sparse cluster entity sits on its own row in this fixture.
    assert len(row_ids) == 6


def test_column_construction_unaffected_by_evidence_change():
    result = _result(*_two_cluster_population())
    spatial = build_spatial_representation(result)
    for region in spatial.regions:
        # Each region here is a single vertical column of three entities.
        assert len(region.columns) == 1


def test_read_order_unaffected_by_evidence_change():
    result = _result(*_two_cluster_population())
    spatial = build_spatial_representation(result)
    assert spatial.read_order == tuple(
        entity.canonical_id for entity in result.entities
    )


def test_evidence_ordering_is_deterministic():
    result = _result(*_two_cluster_population())
    first = build_spatial_evidence(result)
    second = build_spatial_evidence(result)
    assert first == second
    ids = [
        (item.first_entity_id, item.second_entity_id)
        for item in first.linked_pair_evidence
    ]
    assert ids == sorted(ids)


def test_repeated_execution_is_identical():
    result = _result(*_two_cluster_population())
    runs = [build_spatial_evidence(result) for _ in range(5)]
    assert all(run == runs[0] for run in runs)


def test_reconciliation_result_is_not_mutated():
    result = _result(*_two_cluster_population())
    entities_before = tuple(result.entities)
    build_spatial_representation(result)
    build_spatial_evidence(result)
    assert result.entities == entities_before
    assert result.entities is entities_before or result.entities == entities_before


def test_single_region_population_still_works():
    """Regression guard for the common real-card case observed so far
    (Kulin/Gaurav/Satish all currently resolve to a single region): with
    only one component, the fix must produce identical behavior to before,
    since component == full population in that case."""
    entities = tuple(_entity(n, (n * 15, 0, n * 15 + 10, 10)) for n in range(1, 6))
    result = _result(*entities)
    spatial = build_spatial_representation(result)
    assert len(spatial.regions) == 1

    evidence = build_spatial_evidence(result)
    _, full_population_evidence = _region_linkage_graph(entities)
    expected = {
        (item.first_entity_id, item.second_entity_id, item.scale)
        for item in full_population_evidence
        if item.linked
    }
    actual = {
        (item.first_entity_id, item.second_entity_id, item.scale)
        for item in evidence.linked_pair_evidence
    }
    assert actual == expected


def test_transitive_chain_still_collapses_to_one_region():
    """Existing transitive-chain behavior (A-B-C-D collapsing into one
    connected component when each consecutive pair is close) must remain
    unchanged by this provenance-only fix -- this fix does not touch
    _region_linked, _region_linkage_graph's linkage decision, or the
    existing 6.0 / 1.5 constants."""
    chain = tuple(_entity(n, (n * 12, 0, n * 12 + 10, 10)) for n in range(1, 9))
    result = _result(*chain)
    spatial = build_spatial_representation(result)
    assert len(spatial.regions) == 1
