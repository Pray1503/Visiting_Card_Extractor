from copy import deepcopy

from v21 import (
    V21CanonicalEntity,
    V21ContactCandidate,
    V21ReconciliationResult,
    V21SubregionHypothesisResult,
    build_spatial_evidence,
    build_spatial_representation,
    generate_identity_bindings,
    generate_subregion_hypotheses,
)


def _entity(number, text, bbox):
    token_id = f"token-{number:06d}"
    return V21CanonicalEntity(
        canonical_id=f"canonical-{number:06d}",
        source_token_ids=(token_id,),
        pass_ids=("pass_1",),
        canonical_bbox=bbox,
        representative_text=text,
        representative_token_id=token_id,
    )


def _inputs(entities, candidates=()):
    reconciliation = V21ReconciliationResult(entities=tuple(entities))
    spatial = build_spatial_representation(reconciliation)
    evidence = build_spatial_evidence(reconciliation)
    hypotheses = generate_subregion_hypotheses(
        reconciliation, spatial, evidence, tuple(candidates)
    )
    return reconciliation, spatial, evidence, tuple(candidates), hypotheses


def test_phase6_creates_one_binding_per_hypothesis():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Engineer", (0, 30, 60, 50)),
        _entity(3, "Contoso", (0, 60, 80, 90)),
    ]
    _, _, _, _, hypotheses = _inputs(entities)
    result = generate_identity_bindings(*_inputs(entities)[:4], hypotheses)
    assert len(result.bindings) == len(hypotheses.hypotheses)
    assert {item.hypothesis_id for item in result.bindings} == {
        item.hypothesis_id for item in hypotheses.hypotheses
    }


def test_phase6_atomic_candidates_are_nonexclusive_and_structural():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Engineer", (0, 30, 60, 50)),
        _entity(3, "Contoso", (0, 60, 80, 90)),
    ]
    inputs = _inputs(entities)
    result = generate_identity_bindings(*inputs[:4], inputs[4])
    fields = {candidate.field_type for candidate in result.candidates}
    assert fields == {"NAME", "TITLE", "COMPANY"}
    assert any(candidate.text == "Alicia" for candidate in result.candidates)
    assert any(candidate.text == "Engineer" for candidate in result.candidates)
    assert any(candidate.text == "Contoso" for candidate in result.candidates)
    assert all(
        candidate.composition_type in {"ATOMIC", "HORIZONTAL", "VERTICAL"}
        for candidate in result.candidates
    )


def test_phase6_scoring_is_additive_and_weighted_to_one():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Engineer", (0, 30, 60, 50)),
    ]
    inputs = _inputs(entities)
    result = generate_identity_bindings(*inputs[:4], inputs[4])
    for field_type in ("NAME", "TITLE", "COMPANY"):
        pool = tuple(
            candidate
            for candidate in result.candidates
            if candidate.field_type == field_type
        )
        assert pool
        for candidate in pool:
            assert abs(sum(weight for _, weight in candidate.weights) - 1.0) < 1e-9
            assert (
                abs(
                    candidate.score
                    - sum(
                        weight * value
                        for (_, weight), (_, value) in zip(
                            candidate.weights, candidate.normalized_features
                        )
                    )
                )
                < 1e-6
            )


def test_phase6_binding_is_hypothesis_isolated_and_immutable():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Engineer", (0, 30, 60, 50)),
        _entity(3, "Contoso", (0, 60, 80, 90)),
    ]
    reconciliation, spatial, evidence, candidates, hypotheses = _inputs(entities)
    before = deepcopy((reconciliation, spatial, evidence, candidates, hypotheses))
    answer = generate_identity_bindings(
        reconciliation, spatial, evidence, candidates, hypotheses
    )
    assert (reconciliation, spatial, evidence, candidates, hypotheses) == before
    assert len(answer.bindings) == len(hypotheses.hypotheses)
    for binding in answer.bindings:
        assert binding.hypothesis_id in {
            item.hypothesis_id for item in hypotheses.hypotheses
        }
        for field in (
            "name_candidate_id",
            "title_candidate_id",
            "company_candidate_id",
        ):
            candidate_id = getattr(binding, field)
            if candidate_id is not None:
                assert any(
                    candidate.candidate_id == candidate_id
                    for candidate in answer.candidates
                )
                assert any(
                    candidate.hypothesis_id == binding.hypothesis_id
                    for candidate in answer.candidates
                    if candidate.candidate_id == candidate_id
                )


def test_phase6_triplet_composite_is_not_duplicated():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Marie", (45, 0, 85, 20)),
        _entity(3, "Smith", (90, 0, 130, 20)),
    ]

    inputs = _inputs(entities)
    result = generate_identity_bindings(*inputs[:4], inputs[4])

    triplets = [
        candidate
        for candidate in result.candidates
        if candidate.composition_type in {"HORIZONTAL", "VERTICAL"}
        and len(candidate.source_entity_ids) == 3
    ]

    signatures = [
        (candidate.field_type, candidate.source_entity_ids) for candidate in triplets
    ]

    assert len(signatures) == len(set(signatures))


def test_phase6_binding_preserves_valid_multi_field_coverage():
    entities = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Engineer", (0, 30, 60, 50)),
        _entity(3, "Contoso", (0, 60, 80, 90)),
    ]

    inputs = _inputs(entities)
    result = generate_identity_bindings(*inputs[:4], inputs[4])

    for binding in result.bindings:
        selected = {
            "NAME": binding.name_candidate_id,
            "TITLE": binding.title_candidate_id,
            "COMPANY": binding.company_candidate_id,
        }

        # Every selected candidate must be unique across fields.
        selected_ids = [
            candidate_id for candidate_id in selected.values() if candidate_id
        ]
        assert len(selected_ids) == len(set(selected_ids))

        # If all three field types have candidates in this hypothesis,
        # binding should be capable of selecting all three.
        hypothesis_candidates = [
            candidate
            for candidate in result.candidates
            if candidate.hypothesis_id == binding.hypothesis_id
        ]

        available_fields = {candidate.field_type for candidate in hypothesis_candidates}

        for field_type in ("NAME", "TITLE", "COMPANY"):
            if field_type in available_fields:
                assert selected[field_type] is not None


def test_phase6_scale_invariance():
    entities_small = [
        _entity(1, "Alicia", (0, 0, 40, 20)),
        _entity(2, "Marie", (45, 0, 85, 20)),
        _entity(3, "Smith", (90, 0, 130, 20)),
    ]

    entities_large = [
        _entity(1, "Alicia", (0, 0, 400, 200)),
        _entity(2, "Marie", (450, 0, 850, 200)),
        _entity(3, "Smith", (900, 0, 1300, 200)),
    ]

    small_inputs = _inputs(entities_small)
    large_inputs = _inputs(entities_large)

    small_result = generate_identity_bindings(*small_inputs[:4], small_inputs[4])
    large_result = generate_identity_bindings(*large_inputs[:4], large_inputs[4])

    def signatures(result):
        return {
            (
                candidate.field_type,
                candidate.composition_type,
                len(candidate.source_entity_ids),
            )
            for candidate in result.candidates
        }

    assert signatures(small_result) == signatures(large_result)
