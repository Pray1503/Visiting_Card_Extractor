from v21 import (
    V21CanonicalEntity,
    V21ContactCandidate,
    V21ReconciliationResult,
    build_spatial_representation,
    determine_panel_primacy,
)


def _entity(number, text, bbox):
    return V21CanonicalEntity(
        canonical_id=f"canonical-{number:06d}",
        source_token_ids=(f"token-{number:06d}",),
        pass_ids=("pass_1",),
        canonical_bbox=bbox,
        representative_text=text,
        representative_token_id=f"token-{number:06d}",
    )


def _candidate(number, field_type, entity_id):
    return V21ContactCandidate(
        candidate_id=f"candidate-{number:06d}",
        field_type=field_type,
        normalized_value=f"value-{number}",
        source_canonical_entity_ids=(entity_id,),
    )


def _inputs(entities, candidates=()):
    reconciliation = V21ReconciliationResult(entities=tuple(entities))
    spatial = build_spatial_representation(reconciliation)
    return reconciliation, spatial, tuple(candidates)


def test_single_panel_is_primary():
    reconciliation, spatial, candidates = _inputs(
        [_entity(1, "Avery Stone", (0, 0, 80, 20))]
    )
    result = determine_panel_primacy(reconciliation, spatial, candidates)
    assert result.primary_region_id == "region-000001"
    assert result.ranked_regions[0].rank == 1


def test_identity_contact_structure_outranks_catalog_structure():
    reconciliation, spatial, candidates = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 40)),
            _entity(2, "Design Lead", (0, 50, 80, 70)),
            _entity(3, "shop@example.org", (0, 80, 120, 95)),
            _entity(4, "SERIES 400 2024", (1000, 0, 620, 20)),
            _entity(5, "MODEL SPECIFICATION", (1000, 30, 700, 50)),
            _entity(6, "PACK 12", (1000, 60, 580, 78)),
        ],
        [_candidate(1, "EMAIL", "canonical-000003")],
    )
    result = determine_panel_primacy(reconciliation, spatial, candidates)
    assert result.ranked_regions[0].region_id == "region-000001"


def test_contact_density_alone_does_not_win():
    reconciliation, spatial, candidates = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 40)),
            _entity(2, "Design Lead", (0, 50, 80, 70)),
            _entity(3, "x@y.org", (0, 80, 120, 95)),
            _entity(4, "z@q.org", (0, 100, 120, 115)),
            _entity(5, "12-345-6789", (0, 120, 120, 135)),
            _entity(6, "BRAND MARK", (1000, 0, 650, 80)),
            _entity(7, "LONG PRODUCT CATALOG DESCRIPTION", (1000, 90, 760, 120)),
        ],
        [
            _candidate(1, "EMAIL", "canonical-000003"),
            _candidate(2, "EMAIL", "canonical-000004"),
            _candidate(3, "PHONE", "canonical-000005"),
        ],
    )
    result = determine_panel_primacy(reconciliation, spatial, candidates)
    assert result.ranked_regions[0].region_id == "region-000001"


def test_branding_prominence_alone_does_not_win():
    reconciliation, spatial, candidates = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 35)),
            _entity(2, "Design Lead", (0, 45, 90, 65)),
            _entity(3, "m@example.org", (0, 75, 110, 90)),
            _entity(4, "BRAND EMBLEM", (1000, 0, 800, 150)),
            _entity(5, "VISUAL SYSTEM", (1000, 170, 700, 210)),
        ],
        [_candidate(1, "EMAIL", "canonical-000003")],
    )
    result = determine_panel_primacy(reconciliation, spatial, candidates)
    assert result.ranked_regions[0].region_id == "region-000001"


def test_two_legitimate_panels_are_retained_and_ranked():
    reconciliation, spatial, candidates = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 30)),
            _entity(2, "Design Lead", (0, 40, 90, 60)),
            _entity(3, "m@example.org", (0, 70, 110, 85)),
            _entity(4, "Jordan Vale", (1000, 0, 600, 30)),
            _entity(5, "Studio Lead", (1000, 40, 590, 60)),
            _entity(6, "j@example.org", (1000, 70, 610, 85)),
        ],
        [
            _candidate(1, "EMAIL", "canonical-000003"),
            _candidate(2, "EMAIL", "canonical-000006"),
        ],
    )
    result = determine_panel_primacy(reconciliation, spatial, candidates)
    assert len(result.ranked_regions) == 2
    assert {item.region_id for item in result.ranked_regions} == {
        "region-000001",
        "region-000002",
    }


def test_ties_use_deterministic_region_order():
    reconciliation, spatial, candidates = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 30)),
            _entity(2, "Jordan Vale", (1000, 0, 600, 30)),
        ]
    )
    first = determine_panel_primacy(reconciliation, spatial, candidates)
    second = determine_panel_primacy(reconciliation, spatial, candidates)
    assert first == second
    assert [item.region_id for item in first.ranked_regions] == [
        "region-000001",
        "region-000002",
    ]


def test_equivalent_scaled_layout_preserves_ranking():
    small = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 100, 30)),
            _entity(2, "m@example.org", (0, 40, 110, 55)),
            _entity(3, "BRAND MARK", (1000, 0, 650, 80)),
        ],
        [_candidate(1, "EMAIL", "canonical-000002")],
    )
    large = _inputs(
        [
            _entity(1, "Avery Stone", (0, 0, 200, 60)),
            _entity(2, "m@example.org", (0, 80, 220, 110)),
            _entity(3, "BRAND MARK", (1000, 0, 1300, 160)),
        ],
        [_candidate(1, "EMAIL", "canonical-000002")],
    )
    assert (
        determine_panel_primacy(*small).primary_region_id
        == determine_panel_primacy(*large).primary_region_id
    )


def test_phase_inputs_are_not_mutated():
    reconciliation, spatial, candidates = _inputs(
        [_entity(1, "Avery Stone", (0, 0, 100, 30))]
    )
    before = (reconciliation, spatial, candidates)
    determine_panel_primacy(reconciliation, spatial, candidates)
    assert (reconciliation, spatial, candidates) == before


def test_repeated_execution_is_identical():
    reconciliation, spatial, candidates = _inputs(
        [_entity(1, "Avery Stone", (0, 0, 100, 30))]
    )
    assert determine_panel_primacy(
        reconciliation, spatial, candidates
    ) == determine_panel_primacy(reconciliation, spatial, candidates)
