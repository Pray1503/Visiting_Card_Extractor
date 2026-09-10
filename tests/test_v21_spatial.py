from v21 import (
    V21CanonicalEntity,
    V21ReconciliationResult,
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


def test_single_panel_single_column():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (0, 30, 20, 40)),
            _entity(3, (0, 60, 20, 70)),
        )
    )
    assert len(result.regions) == 1
    assert len(result.regions[0].columns) == 1
    assert [row.entity_ids for row in result.regions[0].rows] == [
        ("canonical-000001",),
        ("canonical-000002",),
        ("canonical-000003",),
    ]


def test_two_columns_keep_rows_and_columns_as_ids():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (100, 0, 120, 10)),
            _entity(3, (0, 30, 20, 40)),
            _entity(4, (100, 30, 120, 40)),
        )
    )
    assert len(result.regions) == 1
    assert len(result.regions[0].columns) == 2
    assert [row.entity_ids for row in result.regions[0].rows] == [
        ("canonical-000001", "canonical-000002"),
        ("canonical-000003", "canonical-000004"),
    ]


def test_two_separated_regions_do_not_share_rows():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (0, 30, 20, 40)),
            _entity(3, (300, 300, 320, 310)),
        )
    )
    assert len(result.regions) == 2
    assert [len(region.rows) for region in result.regions] == [2, 1]


def test_panel_relative_coordinates_preserve_global_geometry():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 20, 20)), _entity(2, (100, 100, 120, 120)))
    )
    geometry = result.geometry_for("canonical-000001")
    assert geometry.global_bbox == (0, 0, 20, 20)
    assert geometry.relative_x == 0.0
    assert geometry.relative_y == 0.0


def test_irregular_entities_remain_present_and_output_is_deterministic():
    result_input = _result(_entity(1, (0, 0, 10, 10)), _entity(2, (200, 200, 210, 210)))
    first = build_spatial_representation(result_input)
    second = build_spatial_representation(result_input)
    assert first == second
    assert set(first.read_order) == {"canonical-000001", "canonical-000002"}


def test_non_destructive_phase_two_result():
    phase_two = _result(_entity(1, (10, 10, 20, 20)))
    before = phase_two
    build_spatial_representation(phase_two)
    assert phase_two == before


def test_same_row_uses_vertical_alignment_not_x_position():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (200, 2, 230, 12)),
        )
    )
    assert len(result.regions[0].rows) == 1


def test_different_rows_remain_separate_with_similar_x_positions():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (2, 30, 22, 40)),
        )
    )
    assert len(result.regions[0].rows) == 2


def test_wide_entity_does_not_inflate_vertical_row_tolerance():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 1000, 10)),
            _entity(2, (0, 30, 20, 40)),
            _entity(3, (100, 30, 120, 40)),
        )
    )
    assert [row.entity_ids for row in result.regions[0].rows] == [
        ("canonical-000001",),
        ("canonical-000002", "canonical-000003"),
    ]


def test_row_grouping_is_scale_invariant():
    small = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (40, 2, 60, 12)),
            _entity(3, (0, 30, 20, 40)),
        )
    )
    large = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 40, 20)),
            _entity(2, (80, 4, 120, 24)),
            _entity(3, (0, 60, 40, 80)),
        )
    )
    assert [row.entity_ids for row in small.regions[0].rows] == [
        ("canonical-000001", "canonical-000002"),
        ("canonical-000003",),
    ]
    assert [row.entity_ids for row in large.regions[0].rows] == [
        ("canonical-000001", "canonical-000002"),
        ("canonical-000003",),
    ]


def test_same_row_allows_height_variation_with_overlap():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 20)),
            _entity(2, (40, 5, 60, 15)),
        )
    )
    assert len(result.regions[0].rows) == 1


def test_non_overlapping_lines_do_not_collapse():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 20, 10)),
            _entity(2, (10, 25, 30, 35)),
        )
    )
    assert len(result.regions[0].rows) == 2


def test_similar_heights_with_strong_overlap_share_row():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 20, 20)), _entity(2, (40, 3, 60, 23)))
    )
    assert len(result.regions[0].rows) == 1


def test_moderately_different_heights_with_overlap_share_row():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 20, 30)), _entity(2, (40, 8, 60, 23)))
    )
    assert len(result.regions[0].rows) == 1


def test_partial_overlap_with_aligned_centers_shares_row():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 20, 20)), _entity(2, (40, 10, 60, 30)))
    )
    assert len(result.regions[0].rows) == 1


def test_tall_containment_with_displaced_short_entity_is_separate():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 40, 100)), _entity(2, (50, 70, 70, 90)))
    )
    assert len(result.regions[0].rows) == 2


def test_substantial_overlap_with_height_disparity_is_separate():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 40, 90)), _entity(2, (50, 60, 70, 80)))
    )
    assert len(result.regions[0].rows) == 2


def test_phone_like_alignment_survives_height_difference():
    result = build_spatial_representation(
        _result(_entity(1, (0, 0, 250, 48)), _entity(2, (250, -2, 340, 41)))
    )
    assert len(result.regions[0].rows) == 1


def test_tall_entity_does_not_chain_displaced_entities_into_one_row():
    result = build_spatial_representation(
        _result(
            _entity(1, (0, 0, 40, 100)),
            _entity(2, (50, 20, 70, 40)),
            _entity(3, (80, 70, 100, 90)),
        )
    )
    assert [row.entity_ids for row in result.regions[0].rows] == [
        ("canonical-000001", "canonical-000002"),
        ("canonical-000003",),
    ]
