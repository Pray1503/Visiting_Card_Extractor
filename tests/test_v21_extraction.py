from v21 import (
    V21CanonicalEntity,
    V21ReconciliationResult,
    build_spatial_representation,
    extract_deterministic_candidates,
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


def _spatial(*entities):
    reconciliation = V21ReconciliationResult(entities=tuple(entities))
    return reconciliation, build_spatial_representation(reconciliation)


def _values(reconciliation, spatial):
    return extract_deterministic_candidates(reconciliation, spatial)


def test_basic_email_extraction():
    reconciliation, spatial = _spatial(_entity(1, "Sales@Example.COM", (0, 0, 100, 20)))
    candidates = _values(reconciliation, spatial)
    assert [(item.field_type, item.normalized_value) for item in candidates] == [
        ("EMAIL", "sales@example.com")
    ]


def test_multiple_emails_are_retained():
    reconciliation, spatial = _spatial(
        _entity(1, "sales@example.com", (0, 0, 100, 20)),
        _entity(2, "pune@example.com", (0, 30, 100, 50)),
    )
    assert [item.normalized_value for item in _values(reconciliation, spatial)] == [
        "sales@example.com",
        "pune@example.com",
    ]


def test_email_normalization_does_not_mutate_raw_canonical_text():
    entity = _entity(1, "<Sales@Example.COM>,", (0, 0, 100, 20))
    reconciliation, spatial = _spatial(entity)
    _values(reconciliation, spatial)
    assert entity.representative_text == "<Sales@Example.COM>,"


def test_obvious_at_false_positive_is_rejected():
    reconciliation, spatial = _spatial(_entity(1, "contact @ office", (0, 0, 100, 20)))
    assert _values(reconciliation, spatial) == ()


def test_basic_website_extraction():
    reconciliation, spatial = _spatial(
        _entity(1, "https://Example.com", (0, 0, 100, 20))
    )
    candidates = _values(reconciliation, spatial)
    assert [(item.field_type, item.normalized_value) for item in candidates] == [
        ("WEBSITE", "https://example.com")
    ]


def test_multiple_websites_are_retained():
    reconciliation, spatial = _spatial(
        _entity(1, "www.example.com", (0, 0, 100, 20)),
        _entity(2, "example.org", (0, 30, 100, 50)),
    )
    assert [item.normalized_value for item in _values(reconciliation, spatial)] == [
        "www.example.com",
        "example.org",
    ]


def test_website_normalization_is_deterministic():
    reconciliation, spatial = _spatial(_entity(1, "WWW.Example.COM,", (0, 0, 100, 20)))
    first = _values(reconciliation, spatial)
    second = _values(reconciliation, spatial)
    assert first == second
    assert first[0].normalized_value == "www.example.com"


def test_basic_phone_extraction():
    reconciliation, spatial = _spatial(_entity(1, "+91 22 4355 5555", (0, 0, 150, 20)))
    candidates = _values(reconciliation, spatial)
    assert any(item.field_type == "PHONE" for item in candidates)


def test_varied_phone_formats_are_retained():
    reconciliation, spatial = _spatial(
        _entity(1, "+1 212 555 1234", (0, 0, 120, 20)),
        _entity(2, "022-12345678", (0, 30, 120, 50)),
    )
    phones = [
        item.normalized_value
        for item in _values(reconciliation, spatial)
        if item.field_type == "PHONE"
    ]
    assert phones[:2] == ["+1 212 555 1234", "022-12345678"]


def test_split_phone_fragments_can_be_reconstructed():
    reconciliation, spatial = _spatial(
        _entity(1, "91-22-4355", (0, 0, 90, 20)),
        _entity(2, "5555", (92, 0, 140, 20)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(
        item.source_canonical_entity_ids == ("canonical-000001", "canonical-000002")
        for item in phones
    )


def test_split_phone_zero_gap_is_reconstructed():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (30, 0, 58, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_split_phone_small_box_overlap_is_reconstructed():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (28, 0, 58, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_large_box_overlap_is_rejected():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (15, 0, 45, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_reversed_fragment_order_is_rejected():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (40, 0, 70, 10)),
        _entity(2, "6789", (0, 0, 30, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_small_overlap_reconstruction_is_scale_invariant():
    small = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (28, 0, 58, 10)),
    )
    large = _spatial(
        _entity(1, "12-345", (0, 0, 60, 20)),
        _entity(2, "6789", (56, 0, 116, 20)),
    )
    assert any(
        len(item.source_canonical_entity_ids) == 2
        for item in _values(*small)
        if item.field_type == "PHONE"
    )
    assert any(
        len(item.source_canonical_entity_ids) == 2
        for item in _values(*large)
        if item.field_type == "PHONE"
    )


def test_isolated_short_numbers_are_rejected():
    reconciliation, spatial = _spatial(
        _entity(1, "9", (0, 0, 20, 20)),
        _entity(2, "72", (0, 30, 20, 50)),
        _entity(3, "400", (0, 60, 30, 80)),
        _entity(4, "004", (0, 90, 30, 110)),
    )
    assert not any(
        item.field_type == "PHONE" for item in _values(reconciliation, spatial)
    )


def test_social_url_extraction():
    reconciliation, spatial = _spatial(
        _entity(1, "linkedin.com/in/example", (0, 0, 160, 20))
    )
    candidates = _values(reconciliation, spatial)
    assert [(item.field_type, item.normalized_value) for item in candidates] == [
        ("SOCIAL", "linkedin.com/in/example")
    ]


def test_email_is_not_classified_as_social_handle():
    reconciliation, spatial = _spatial(
        _entity(1, "person@example.com", (0, 0, 150, 20))
    )
    candidates = _values(reconciliation, spatial)
    assert [item.field_type for item in candidates] == ["EMAIL"]


def test_handle_requires_structural_context():
    reconciliation, spatial = _spatial(_entity(1, "@example", (0, 0, 100, 20)))
    candidates = _values(reconciliation, spatial)
    assert [(item.field_type, item.normalized_value) for item in candidates] == [
        ("SOCIAL", "@example")
    ]


def test_provenance_points_to_canonical_entity_ids():
    reconciliation, spatial = _spatial(_entity(7, "sales@example.com", (0, 0, 120, 20)))
    candidate = _values(reconciliation, spatial)[0]
    assert candidate.source_canonical_entity_ids == ("canonical-000007",)


def test_candidate_ordering_and_repeated_execution_are_deterministic():
    reconciliation, spatial = _spatial(
        _entity(1, "sales@example.com", (0, 0, 120, 20)),
        _entity(2, "www.example.com", (0, 30, 120, 50)),
        _entity(3, "+1 212 555 1234", (0, 60, 120, 80)),
    )
    first = _values(reconciliation, spatial)
    second = _values(reconciliation, spatial)
    assert first == second
    assert [item.candidate_id for item in first] == [
        "candidate-000001",
        "candidate-000002",
        "candidate-000003",
    ]


def test_phase_two_three_invariants_and_non_destructive_input():
    reconciliation, spatial = _spatial(
        _entity(1, "sales@example.com", (0, 0, 120, 20)),
        _entity(2, "www.example.com", (0, 30, 120, 50)),
    )
    before_reconciliation = reconciliation
    before_spatial = spatial
    _values(reconciliation, spatial)
    assert reconciliation == before_reconciliation
    assert spatial == before_spatial


def test_split_phone_small_normalized_gap():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (32, 0, 58, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(item.normalized_value == "12-3456789" for item in phones)


def test_split_phone_with_left_delimiter():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345/", (0, 0, 35, 12)),
        _entity(2, "678", (37, 0, 58, 12)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(
        item.normalized_value == "12-345/678"
        and item.source_canonical_entity_ids
        == (
            "canonical-000001",
            "canonical-000002",
        )
        for item in phones
    )


def test_split_phone_scales_with_entity_height():
    small = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (32, 0, 58, 10)),
    )
    large = _spatial(
        _entity(1, "12-345", (0, 0, 60, 20)),
        _entity(2, "6789", (64, 0, 116, 20)),
    )
    small_values = [
        item.normalized_value for item in _values(*small) if item.field_type == "PHONE"
    ]
    large_values = [
        item.normalized_value for item in _values(*large) if item.field_type == "PHONE"
    ]
    assert "12-3456789" in small_values
    assert "12-3456789" in large_values


def test_split_phone_allows_different_fragment_lengths():
    reconciliation, spatial = _spatial(
        _entity(1, "+12-3456", (0, 0, 42, 10)),
        _entity(2, "789", (44, 0, 64, 10)),
    )
    phones = [
        item.normalized_value
        for item in _values(reconciliation, spatial)
        if item.field_type == "PHONE"
    ]
    assert "+12-3456789" in phones


def test_split_phone_country_neutral_structure():
    reconciliation, spatial = _spatial(
        _entity(1, "45.678", (0, 0, 34, 10)),
        _entity(2, "90123", (36, 0, 70, 10)),
    )
    phones = [
        item.normalized_value
        for item in _values(reconciliation, spatial)
        if item.field_type == "PHONE"
    ]
    assert "45.67890123" in phones


def test_complete_phones_on_same_row_remain_separate():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345-6789", (0, 0, 55, 10)),
        _entity(2, "98-765-4321", (57, 0, 112, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert [item.source_canonical_entity_ids for item in phones] == [
        ("canonical-000001",),
        ("canonical-000002",),
    ]


def test_large_gap_rejects_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (60, 0, 86, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_different_rows_reject_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (0, 30, 26, 40)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_intervening_text_rejects_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "Office", (32, 0, 70, 10)),
        _entity(3, "6789", (72, 0, 98, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_column_boundary_rejects_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 20, 10)),
        _entity(2, "6789", (100, 0, 126, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_postal_like_fragments_reject_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12345", (0, 0, 32, 10)),
        _entity(2, "678", (34, 0, 56, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert not any(len(item.source_canonical_entity_ids) == 2 for item in phones)


def test_year_date_rejects_phone_candidate():
    reconciliation, spatial = _spatial(_entity(1, "2024-01-01", (0, 0, 90, 10)))
    assert not any(
        item.field_type == "PHONE" for item in _values(reconciliation, spatial)
    )


def test_apartment_numbers_reject_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "12", (0, 0, 18, 10)),
        _entity(2, "345", (20, 0, 42, 10)),
    )
    assert not any(
        item.field_type == "PHONE" for item in _values(reconciliation, spatial)
    )


def test_catalog_numbers_reject_reconstruction():
    reconciliation, spatial = _spatial(
        _entity(1, "123", (0, 0, 22, 10)),
        _entity(2, "456", (24, 0, 46, 10)),
    )
    assert not any(
        item.field_type == "PHONE" for item in _values(reconciliation, spatial)
    )


def test_reconstructed_candidate_is_additional_and_provenant():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (32, 0, 58, 10)),
    )
    phones = [
        item for item in _values(reconciliation, spatial) if item.field_type == "PHONE"
    ]
    assert any(
        item.source_canonical_entity_ids == ("canonical-000001", "canonical-000002")
        for item in phones
    )
    assert reconciliation.entities[0].representative_text == "12-345"
    assert reconciliation.entities[1].representative_text == "6789"


def test_split_phone_output_is_repeatable():
    reconciliation, spatial = _spatial(
        _entity(1, "12-345", (0, 0, 30, 10)),
        _entity(2, "6789", (32, 0, 58, 10)),
    )
    assert _values(reconciliation, spatial) == _values(reconciliation, spatial)
