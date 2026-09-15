from copy import deepcopy

from v21 import (
    V21ContactCandidate,
    V21CanonicalEntity,
    V21ReconciliationResult,
    V21_MAX_CUT_SETS,
    V21_MAX_HYPOTHESES_PER_REGION,
    V21_MAX_OUTLIER_EDGES,
    V21_MAX_TOTAL_HYPOTHESES,
    build_spatial_evidence,
    build_spatial_representation,
    generate_subregion_hypotheses,
)


def _entity(number, bbox, text=None):
    token_id = f"token-{number:04d}"
    return V21CanonicalEntity(
        canonical_id=f"canonical-{number:04d}",
        source_token_ids=(token_id,),
        pass_ids=("pass_1",),
        canonical_bbox=tuple(float(value) for value in bbox),
        representative_text=text or f"entity-{number}",
        representative_token_id=token_id,
    )


def _inputs(entities, candidates=()):
    reconciliation = V21ReconciliationResult(entities=tuple(entities))
    spatial = build_spatial_representation(reconciliation)
    evidence = build_spatial_evidence(reconciliation)
    return reconciliation, spatial, evidence, tuple(candidates)


def _run(entities, candidates=()):
    return generate_subregion_hypotheses(*_inputs(entities, candidates))


def _line(start, positions, scale=1.0, offset=0):
    return [
        _entity(
            offset + index,
            (start + position * scale, 0, start + (position + 1) * scale, scale),
        )
        for index, position in enumerate(positions, 1)
    ]


def _split_hypotheses(result):
    return [
        hypothesis for hypothesis in result.hypotheses if len(hypothesis.groups) > 1
    ]


def test_clear_two_cluster_separation_produces_h0_and_spatial_split():
    result = _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))
    assert any(len(hypothesis.groups) == 1 for hypothesis in result.hypotheses)
    assert any(
        sorted(len(group.entity_ids) for group in hypothesis.groups) == [3, 3]
        for hypothesis in _split_hypotheses(result)
    )


def test_uniform_distribution_keeps_h0_valid():
    result = _run(_line(0, range(8)))
    assert any(len(hypothesis.groups) == 1 for hypothesis in result.hypotheses)


def test_clear_internal_gap_is_scale_normalized():
    entities = _line(0, (0, 2, 4, 8, 10, 12))
    result = _run(entities)
    split = _split_hypotheses(result)
    assert split
    expected_ids = {entity.canonical_id for entity in entities}
    assert any(
        {entity_id for group in hypothesis.groups for entity_id in group.entity_ids}
        == expected_ids
        for hypothesis in split
    )


def test_uniform_spacing_does_not_split_on_skip_edges():
    result = _run(_line(0, (0, 2, 4, 6, 8, 10)))
    assert not _split_hypotheses(result)


def test_multiple_phase3_regions_remain_isolated():
    entities = _line(0, (0, 2, 4)) + _line(100, (0, 2, 4), offset=3)
    result = _run(entities)
    region_ids = {hypothesis.region_id for hypothesis in result.hypotheses}
    assert len(region_ids) == 2
    for hypothesis in result.hypotheses:
        region_entities = {
            entity_id for group in hypothesis.groups for entity_id in group.entity_ids
        }
        region = next(
            item
            for item in _inputs(entities)[1].regions
            if item.region_id == hypothesis.region_id
        )
        assert region_entities <= set(region.entity_ids)


def test_uniform_coordinate_scaling_preserves_structure():
    small = _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))
    large_entities = _line(0, (0, 2, 4), scale=10) + _line(
        90, (0, 2, 4), scale=10, offset=3
    )
    large = _run(large_entities)

    def shapes(result):
        return sorted(
            sorted(len(group.entity_ids) for group in hypothesis.groups)
            for hypothesis in result.hypotheses
        )

    assert shapes(small) == shapes(large)


def test_repeated_execution_is_deterministic():
    entities = _line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3)
    first = _run(entities)
    assert first == _run(entities)
    assert first == _run(entities)


def test_inputs_are_immutable():
    entities = _line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3)
    inputs = _inputs(entities)
    before = deepcopy(inputs)
    generate_subregion_hypotheses(*inputs)
    assert inputs == before


def test_contact_anchor_cannot_manufacture_split():
    entities = _line(0, range(6))
    candidate = V21ContactCandidate(
        "candidate-anchor", "EMAIL", "anchor@example.test", (entities[0].canonical_id,)
    )
    result = _run(entities, (candidate,))
    assert not _split_hypotheses(result)


def test_zero_one_and_two_entity_regions_are_safe():
    assert _run([]).hypotheses == ()
    for count in (1, 2):
        result = _run(_line(0, range(count)))
        assert result.hypotheses
        assert all(len(hypothesis.groups) == 1 for hypothesis in result.hypotheses)


def test_no_contact_candidate_region_is_supported():
    result = _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))
    assert result.hypotheses


def test_ambiguous_boundary_retains_competing_deterministic_options():
    result = _run(_line(0, (0, 2, 4, 6, 8)))
    assert result.hypotheses
    assert result == _run(_line(0, (0, 2, 4, 6, 8)))


def test_sparse_edge_population_uses_deterministic_fallback():
    result = _run(_line(0, (0, 2, 4)))
    assert result.hypotheses
    assert result == _run(_line(0, (0, 2, 4)))


def test_equal_scores_use_deterministic_tie_breaking():
    result = _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))
    ordered = [
        (hypothesis.score, hypothesis.hypothesis_id, hypothesis.rank)
        for hypothesis in result.hypotheses
    ]
    assert [item[2] for item in ordered] == list(range(1, len(ordered) + 1))
    assert result == _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))


def test_three_stage_boundedness_is_exposed():
    entities = _line(0, range(32))
    result = _run(entities)
    assert result.diagnostics
    for diagnostic in result.diagnostics:
        assert diagnostic.candidate_outlier_edges_considered <= V21_MAX_OUTLIER_EDGES
        assert diagnostic.candidate_cut_sets_evaluated <= V21_MAX_CUT_SETS
        assert diagnostic.hypotheses_returned <= V21_MAX_HYPOTHESES_PER_REGION
    assert len(result.hypotheses) <= V21_MAX_TOTAL_HYPOTHESES


def test_multi_edge_cut_can_disconnect_redundant_boundary():
    entities = _line(0, (0, 2, 3, 4)) + _line(9, (0, 2, 4), offset=4)
    result = _run(entities)
    assert any(
        len(hypothesis.cut_edge_ids) >= 2 and len(hypothesis.groups) > 1
        for hypothesis in result.hypotheses
    )


def test_singleton_suppression_rejects_one_weak_peripheral_edge():
    peripheral_id = "canonical-0004"
    entities = _line(0, (0, 2, 4)) + [_entity(4, (9, 0, 10, 1))]
    result = _run(entities)
    assert not any(
        any(group.entity_ids == (peripheral_id,) for group in hypothesis.groups)
        for hypothesis in _split_hypotheses(result)
    )


def test_boundedness_stress_does_not_enumerate_edge_power_set():
    result = _run(_line(0, range(48)))
    assert result.diagnostics[0].candidate_cut_sets_evaluated <= V21_MAX_CUT_SETS
    assert len(result.hypotheses) <= V21_MAX_TOTAL_HYPOTHESES


def test_provenance_contains_region_groups_cuts_and_components():
    result = _run(_line(0, (0, 2, 4)) + _line(9, (0, 2, 4), offset=3))
    for hypothesis in result.hypotheses:
        assert hypothesis.region_id
        assert hypothesis.groups
        assert all(group.entity_ids for group in hypothesis.groups)
        assert hypothesis.score_components
        assert hypothesis.hypothesis_id
