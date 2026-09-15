#!/usr/bin/env python3
"""V21 Phase 1: preserve raw two-pass PaddleOCR evidence."""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from itertools import combinations
from math import hypot
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional, Tuple

import cv2
import numpy as np

from core.config import CFG, IMAGE_EXTENSIONS
from imaging import deskew, preprocess
from ocr.paddle import _get_paddle

log = logging.getLogger("VC_OCR_v21")


@dataclass(frozen=True)
class V21OCREvidence:
    token_id: str
    pass_id: str
    raw_text: str
    bbox: Tuple[Tuple[float, float], ...]
    confidence: float
    metadata: Tuple[Tuple[str, Any], ...] = ()


class V21EvidencePool:
    """Append-only storage for raw V21 OCR observations."""

    def __init__(self, pool_id: str = "card") -> None:
        self._pool_id = pool_id
        self._observations: list[V21OCREvidence] = []

    def append(
        self,
        pass_id: str,
        raw_text: str,
        bbox: Iterable[Iterable[float]],
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> V21OCREvidence:
        observation_index = len(self._observations) + 1
        token_id = f"{self._pool_id}-token-{observation_index:06d}"
        observation = V21OCREvidence(
            token_id=token_id,
            pass_id=pass_id,
            raw_text=raw_text,
            bbox=tuple(tuple(float(value) for value in point) for point in bbox),
            confidence=float(confidence),
            metadata=tuple(
                (str(key), value) for key, value in (metadata or {}).items()
            ),
        )
        self._observations.append(observation)
        return observation

    @property
    def observations(self) -> tuple[V21OCREvidence, ...]:
        return tuple(self._observations)

    def by_pass(self, pass_id: str) -> tuple[V21OCREvidence, ...]:
        return tuple(item for item in self._observations if item.pass_id == pass_id)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "token_id": item.token_id,
                "pass_id": item.pass_id,
                "raw_text": item.raw_text,
                "bbox": [list(point) for point in item.bbox],
                "confidence": item.confidence,
                "metadata": dict(item.metadata),
            }
            for item in self._observations
        ]


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class V21CanonicalEntity:
    canonical_id: str
    source_token_ids: Tuple[str, ...]
    pass_ids: Tuple[str, ...]
    canonical_bbox: BBox
    representative_text: str
    representative_token_id: str


@dataclass(frozen=True)
class V21ReconciliationResult:
    entities: Tuple[V21CanonicalEntity, ...]


def _bbox(observation: V21OCREvidence) -> BBox:
    xs = [point[0] for point in observation.bbox]
    ys = [point[1] for point in observation.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: BBox, second: BBox) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _spatially_compatible(first: V21OCREvidence, second: V21OCREvidence) -> bool:
    """Return true only for strong cross-pass correspondence.

    A match requires meaningful area overlap and centers close relative to the
    smaller observation. This prevents identical text in separate regions from
    merging while allowing one pass to split or combine the other pass's box.
    """
    first_box = _bbox(first)
    second_box = _bbox(second)
    first_area = _area(first_box)
    second_area = _area(second_box)
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0.0:
        return False

    intersection = _intersection_area(first_box, second_box)
    overlap_of_smaller = intersection / smaller_area
    union_area = first_area + second_area - intersection
    iou = intersection / union_area if union_area else 0.0
    first_center = (
        (first_box[0] + first_box[2]) / 2,
        (first_box[1] + first_box[3]) / 2,
    )
    second_center = (
        (second_box[0] + second_box[2]) / 2,
        (second_box[1] + second_box[3]) / 2,
    )
    scale = hypot(
        min(first_box[2] - first_box[0], second_box[2] - second_box[0]),
        min(first_box[3] - first_box[1], second_box[3] - second_box[1]),
    )
    normalized_center_distance = (
        hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
        / scale
        if scale > 0.0
        else float("inf")
    )
    return (
        overlap_of_smaller >= 0.50 or iou >= 0.25
    ) and normalized_center_distance <= 1.25


def _union_bbox(observations: Iterable[V21OCREvidence]) -> BBox:
    boxes = [_bbox(observation) for observation in observations]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _representative(
    observations: Iterable[V21OCREvidence], union: BBox
) -> V21OCREvidence:
    union_area = _area(union)

    def ranking(observation: V21OCREvidence) -> tuple[float, float, int, str]:
        coverage = _area(_bbox(observation)) / union_area if union_area else 0.0
        return (
            coverage,
            observation.confidence,
            len(observation.raw_text),
            observation.token_id,
        )

    return max(observations, key=ranking)


def reconcile_evidence(pool: V21EvidencePool) -> V21ReconciliationResult:
    """Build a deterministic canonical view without changing the evidence pool.

    Only observations from different passes can create an edge. Connected
    components therefore reconcile pass-specific echoes and split/combined
    tokenizations, while same-pass observations remain separate by default.
    Components and source IDs are ordered by geometry and stable token ID.
    """
    observations = tuple(pool.observations)
    adjacency = {item.token_id: set() for item in observations}
    for index, first in enumerate(observations):
        for second in observations[index + 1 :]:
            if first.pass_id == second.pass_id:
                continue
            if _spatially_compatible(first, second):
                adjacency[first.token_id].add(second.token_id)
                adjacency[second.token_id].add(first.token_id)

    by_id = {item.token_id: item for item in observations}
    components: list[tuple[V21OCREvidence, ...]] = []
    visited: set[str] = set()
    for observation in sorted(observations, key=lambda item: item.token_id):
        if observation.token_id in visited:
            continue
        pending = [observation.token_id]
        component_ids: list[str] = []
        while pending:
            token_id = pending.pop(0)
            if token_id in visited:
                continue
            visited.add(token_id)
            component_ids.append(token_id)
            pending.extend(sorted(adjacency[token_id] - visited))
        components.append(tuple(by_id[token_id] for token_id in sorted(component_ids)))

    components.sort(
        key=lambda group: (
            _union_bbox(group)[:2],
            tuple(item.token_id for item in group),
        )
    )
    entities = []
    for index, component in enumerate(components, 1):
        canonical_bbox = _union_bbox(component)
        representative = _representative(component, canonical_bbox)
        entities.append(
            V21CanonicalEntity(
                canonical_id=f"canonical-{index:06d}",
                source_token_ids=tuple(item.token_id for item in component),
                pass_ids=tuple(sorted({item.pass_id for item in component})),
                canonical_bbox=canonical_bbox,
                representative_text=representative.raw_text,
                representative_token_id=representative.token_id,
            )
        )
    return V21ReconciliationResult(entities=tuple(entities))


@dataclass(frozen=True)
class V21PanelRelativeGeometry:
    entity_id: str
    global_bbox: BBox
    relative_x: float
    relative_y: float
    relative_width: float
    relative_height: float


@dataclass(frozen=True)
class V21SpatialRow:
    row_id: str
    entity_ids: Tuple[str, ...]
    column_ids: Tuple[str, ...]


@dataclass(frozen=True)
class V21SpatialColumn:
    column_id: str
    entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class V21SpatialRegion:
    region_id: str
    bbox: BBox
    entity_ids: Tuple[str, ...]
    rows: Tuple[V21SpatialRow, ...]
    columns: Tuple[V21SpatialColumn, ...]


@dataclass(frozen=True)
class V21LinkedPairEvidence:
    """Immutable diagnostic snapshot of the geometry evaluated during region linkage.

    This records the exact pairwise measurements already used to decide whether
    two canonical entities are linked, without imposing any panel semantics or
    changing the underlying region logic.
    """

    first_entity_id: str
    second_entity_id: str
    dx: float
    dy: float
    normalized_dx: float
    normalized_dy: float
    close_x: bool
    close_y: bool
    same_row_band: bool
    same_column_band: bool
    scale: float
    first_bbox: BBox
    second_bbox: BBox
    vertical_overlap: float
    horizontal_overlap: float
    linked: bool


@dataclass(frozen=True)
class V21SpatialEvidence:
    """Immutable, diagnostic-only preservation of the existing region-linkage geometry."""

    linked_pair_evidence: Tuple[V21LinkedPairEvidence, ...]


@dataclass(frozen=True)
class V21SpatialRepresentation:
    regions: Tuple[V21SpatialRegion, ...]
    geometries: Tuple[V21PanelRelativeGeometry, ...]
    read_order: Tuple[str, ...]
    entity_region_ids: Tuple[Tuple[str, str], ...]
    entity_row_ids: Tuple[Tuple[str, str], ...]
    entity_column_ids: Tuple[Tuple[str, str], ...]

    def geometry_for(self, entity_id: str) -> V21PanelRelativeGeometry:
        return next(item for item in self.geometries if item.entity_id == entity_id)

    def region_for(self, entity_id: str) -> str:
        return dict(self.entity_region_ids)[entity_id]

    def row_for(self, entity_id: str) -> Optional[str]:
        return dict(self.entity_row_ids).get(entity_id)

    def column_for(self, entity_id: str) -> Optional[str]:
        return dict(self.entity_column_ids).get(entity_id)


def _center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def _height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def _union(entities: tuple[V21CanonicalEntity, ...]) -> BBox:
    return (
        min(entity.canonical_bbox[0] for entity in entities),
        min(entity.canonical_bbox[1] for entity in entities),
        max(entity.canonical_bbox[2] for entity in entities),
        max(entity.canonical_bbox[3] for entity in entities),
    )


def _linked_pair_evidence(
    first: V21CanonicalEntity, second: V21CanonicalEntity, scale: float
) -> V21LinkedPairEvidence:
    first_box = first.canonical_bbox
    second_box = second.canonical_bbox
    first_center = _center(first_box)
    second_center = _center(second_box)
    dx = abs(first_center[0] - second_center[0])
    dy = abs(first_center[1] - second_center[1])
    vertical_overlap = min(first_box[3], second_box[3]) - max(
        first_box[1], second_box[1]
    )
    horizontal_overlap = min(first_box[2], second_box[2]) - max(
        first_box[0], second_box[0]
    )
    close_x = dx <= max(_width(first_box), _width(second_box), scale) * 6.0
    close_y = dy <= max(_height(first_box), _height(second_box), scale) * 6.0
    same_row_band = vertical_overlap > 0.0 or dy <= scale * 1.5
    same_column_band = horizontal_overlap > 0.0 or dx <= scale * 1.5
    linked = (close_x and same_row_band) or (close_y and same_column_band)
    first_id, second_id = sorted((first.canonical_id, second.canonical_id))
    return V21LinkedPairEvidence(
        first_entity_id=first_id,
        second_entity_id=second_id,
        dx=dx,
        dy=dy,
        normalized_dx=dx / scale if scale > 0.0 else 0.0,
        normalized_dy=dy / scale if scale > 0.0 else 0.0,
        close_x=close_x,
        close_y=close_y,
        same_row_band=same_row_band,
        same_column_band=same_column_band,
        scale=scale,
        first_bbox=first_box,
        second_bbox=second_box,
        vertical_overlap=vertical_overlap,
        horizontal_overlap=horizontal_overlap,
        linked=linked,
    )


def _region_linked(
    first: V21CanonicalEntity, second: V21CanonicalEntity, scale: float
) -> bool:
    first_box = first.canonical_bbox
    second_box = second.canonical_bbox
    first_center = _center(first_box)
    second_center = _center(second_box)
    dx = abs(first_center[0] - second_center[0])
    dy = abs(first_center[1] - second_center[1])
    vertical_overlap = min(first_box[3], second_box[3]) - max(
        first_box[1], second_box[1]
    )
    horizontal_overlap = min(first_box[2], second_box[2]) - max(
        first_box[0], second_box[0]
    )
    close_x = dx <= max(_width(first_box), _width(second_box), scale) * 6.0
    close_y = dy <= max(_height(first_box), _height(second_box), scale) * 6.0
    same_row_band = vertical_overlap > 0.0 or dy <= scale * 1.5
    same_column_band = horizontal_overlap > 0.0 or dx <= scale * 1.5
    return (close_x and same_row_band) or (close_y and same_column_band)


def _region_linkage_graph(
    entities: tuple[V21CanonicalEntity, ...],
) -> tuple[dict[str, set[str]], tuple[V21LinkedPairEvidence, ...]]:
    if not entities:
        return {}, ()
    scale = median(
        [
            max(_width(item.canonical_bbox), _height(item.canonical_bbox))
            for item in entities
        ]
    )
    adjacency = {item.canonical_id: set() for item in entities}
    evidence: list[V21LinkedPairEvidence] = []
    for index, first in enumerate(entities):
        for second in entities[index + 1 :]:
            pair_evidence = _linked_pair_evidence(first, second, scale)
            evidence.append(pair_evidence)
            if pair_evidence.linked:
                adjacency[first.canonical_id].add(second.canonical_id)
                adjacency[second.canonical_id].add(first.canonical_id)
    return adjacency, tuple(
        sorted(
            evidence,
            key=lambda item: (item.first_entity_id, item.second_entity_id),
        )
    )


def _region_components(
    entities: tuple[V21CanonicalEntity, ...],
) -> list[tuple[V21CanonicalEntity, ...]]:
    if not entities:
        return []
    adjacency, _ = _region_linkage_graph(entities)
    by_id = {item.canonical_id: item for item in entities}
    components = []
    visited: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_id):
        if entity.canonical_id in visited:
            continue
        pending = [entity.canonical_id]
        component_ids = []
        while pending:
            entity_id = pending.pop(0)
            if entity_id in visited:
                continue
            visited.add(entity_id)
            component_ids.append(entity_id)
            pending.extend(sorted(adjacency[entity_id] - visited))
        components.append(
            tuple(by_id[entity_id] for entity_id in sorted(component_ids))
        )
    return components


def build_spatial_evidence(
    reconciled_result: V21ReconciliationResult,
) -> V21SpatialEvidence:
    """Return ordered, deterministic evidence for the exact existing region-linkage graph.

    This executes a single ``_region_linkage_graph`` call over the full entity
    population -- the same call whose adjacency determines region membership in
    ``_region_components``/``build_spatial_representation``. Filtering that one
    result for ``linked`` pairs is sufficient: only pairs already scored
    ``linked=True`` can ever share a connected component, so no per-component
    recomputation (and therefore no per-component rescaling) is needed.
    """
    _, evidence = _region_linkage_graph(reconciled_result.entities)
    return V21SpatialEvidence(
        linked_pair_evidence=tuple(
            sorted(
                (item for item in evidence if item.linked),
                key=lambda item: (item.first_entity_id, item.second_entity_id),
            )
        )
    )


def _make_rows(
    entities: tuple[V21CanonicalEntity, ...],
    region_id: str,
    column_by_entity: dict[str, str],
) -> tuple[V21SpatialRow, ...]:
    if not entities:
        return ()
    ordered = sorted(
        entities, key=lambda item: (_center(item.canonical_bbox)[1], item.canonical_id)
    )
    rows: list[list[V21CanonicalEntity]] = []
    row_centers: list[float] = []
    row_heights: list[float] = []
    for entity in ordered:
        box = entity.canonical_bbox
        center_y = _center(box)[1]
        entity_height = _height(box)
        matching = [
            index
            for index, row_y in enumerate(row_centers)
            if (
                max(
                    0.0,
                    min(box[3], row_centers[index] + row_heights[index] / 2)
                    - max(box[1], row_centers[index] - row_heights[index] / 2),
                )
                / max(min(entity_height, row_heights[index]), 1.0)
                >= 0.5
                and abs(center_y - row_y)
                / max(min(entity_height, row_heights[index]), 1.0)
                <= 1.0
            )
        ]
        if matching:
            row_index = matching[0]
            rows[row_index].append(entity)
        else:
            rows.append([entity])
            row_centers.append(center_y)
            row_heights.append(entity_height)

    result = []
    for index, row in enumerate(rows, 1):
        row.sort(key=lambda item: (_center(item.canonical_bbox)[0], item.canonical_id))
        result.append(
            V21SpatialRow(
                row_id=f"{region_id}-row-{index:06d}",
                entity_ids=tuple(item.canonical_id for item in row),
                column_ids=tuple(column_by_entity[item.canonical_id] for item in row),
            )
        )
    return tuple(result)


def _make_columns(
    entities: tuple[V21CanonicalEntity, ...], region_id: str
) -> tuple[V21SpatialColumn, ...]:
    if not entities:
        return ()
    scale = median(
        [
            max(_width(item.canonical_bbox), _height(item.canonical_bbox))
            for item in entities
        ]
    )
    ordered = sorted(
        entities, key=lambda item: (_center(item.canonical_bbox)[0], item.canonical_id)
    )
    columns: list[list[V21CanonicalEntity]] = []
    column_anchors: list[float] = []
    for entity in ordered:
        center_x = _center(entity.canonical_bbox)[0]
        matching = [
            index
            for index, column_x in enumerate(column_anchors)
            if abs(center_x - column_x) <= scale * 1.5
        ]
        if matching:
            column_index = min(
                matching,
                key=lambda index: (abs(center_x - column_anchors[index]), index),
            )
            columns[column_index].append(entity)
        else:
            columns.append([entity])
            column_anchors.append(center_x)
    result = []
    for index, column in enumerate(columns, 1):
        column.sort(
            key=lambda item: (_center(item.canonical_bbox)[1], item.canonical_id)
        )
        result.append(
            V21SpatialColumn(
                column_id=f"{region_id}-column-{index:06d}",
                entity_ids=tuple(item.canonical_id for item in column),
            )
        )
    return tuple(result)


def _relative_geometry(
    entity: V21CanonicalEntity, region_box: BBox
) -> V21PanelRelativeGeometry:
    width = max(_width(region_box), 1.0)
    height = max(_height(region_box), 1.0)
    box = entity.canonical_bbox
    return V21PanelRelativeGeometry(
        entity_id=entity.canonical_id,
        global_bbox=box,
        relative_x=(box[0] - region_box[0]) / width,
        relative_y=(box[1] - region_box[1]) / height,
        relative_width=_width(box) / width,
        relative_height=_height(box) / height,
    )


def build_spatial_representation(
    reconciled_result: V21ReconciliationResult,
) -> V21SpatialRepresentation:
    """Build a non-semantic, deterministic spatial view of canonical entities."""
    components = _region_components(reconciled_result.entities)
    components.sort(
        key=lambda group: (
            _union(group)[:2],
            tuple(item.canonical_id for item in group),
        )
    )
    regions = []
    geometries = []
    read_order = []
    entity_regions = []
    entity_rows = []
    entity_columns = []
    for region_index, component in enumerate(components, 1):
        region_id = f"region-{region_index:06d}"
        region_box = _union(component)
        columns = _make_columns(component, region_id)
        column_by_entity = {
            entity_id: column.column_id
            for column in columns
            for entity_id in column.entity_ids
        }
        rows = _make_rows(component, region_id, column_by_entity)
        regions.append(
            V21SpatialRegion(
                region_id,
                region_box,
                tuple(item.canonical_id for item in component),
                rows,
                columns,
            )
        )
        geometries.extend(_relative_geometry(item, region_box) for item in component)
        entity_regions.extend((item.canonical_id, region_id) for item in component)
        row_by_entity = {
            entity_id: row.row_id for row in rows for entity_id in row.entity_ids
        }
        column_by_entity = {
            entity_id: column.column_id
            for column in columns
            for entity_id in column.entity_ids
        }
        entity_rows.extend(sorted(row_by_entity.items()))
        entity_columns.extend(sorted(column_by_entity.items()))
        for row in rows:
            read_order.extend(row.entity_ids)
    return V21SpatialRepresentation(
        regions=tuple(regions),
        geometries=tuple(sorted(geometries, key=lambda item: item.entity_id)),
        read_order=tuple(read_order),
        entity_region_ids=tuple(sorted(entity_regions)),
        entity_row_ids=tuple(sorted(entity_rows)),
        entity_column_ids=tuple(sorted(entity_columns)),
    )


@dataclass(frozen=True)
class V21ContactCandidate:
    candidate_id: str
    field_type: str
    normalized_value: str
    source_canonical_entity_ids: Tuple[str, ...]


_EMAIL_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)+"
    r"(?![A-Za-z0-9._%+\-])",
    re.IGNORECASE,
)
_SOCIAL_URL_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:https?://)?(?:www\.)?"
    r"(?:instagram\.com|linkedin\.com/in|facebook\.com|twitter\.com|x\.com)"
    r"/[A-Za-z0-9._%\-]+"
    r"(?![A-Za-z0-9._%\-])",
    re.IGNORECASE,
)
_WEBSITE_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9@])"
    r"(?:(?:https?://|www\.)[A-Za-z0-9](?:[A-Za-z0-9._%/?#=&+\-]*[A-Za-z0-9_/#=&+\-])?"
    r"|[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z]{2,63})(?:/[A-Za-z0-9._%/?#=&+\-]*)?)"
    r"(?![A-Za-z0-9@])",
    re.IGNORECASE,
)
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)[+]?\d[\d\s().\-]{5,24}\d(?!\d)")
_HANDLE_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])@[A-Za-z0-9_][A-Za-z0-9._\-]{1,49}(?![A-Za-z0-9._\-])"
)


def _candidate_text(entity: V21CanonicalEntity) -> str:
    return entity.representative_text.strip()


def _normalize_email(value: str) -> str:
    return value.strip("<>()[]{}.,;: \t\r\n").lower()


def _normalize_website(value: str) -> str:
    return value.strip("<>()[]{}.,;: \t\r\n").lower()


def _normalize_phone(value: str) -> str:
    value = value.strip("<>[]{}.,;: \t\r\n")
    value = re.sub(r"\s+", " ", value)
    return value


def _phone_is_structurally_valid(value: str) -> bool:
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", value.strip()):
        return False
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return False
    if not re.search(r"[+()\-\s]", value) and len(digits) < 10:
        return False
    return True


def _phone_fragment_text(value: str) -> bool:
    return bool(re.fullmatch(r"[+]?\d[\d()./\-]*", value.strip()))


def _phone_is_independent_complete(value: str) -> bool:
    return _phone_is_structurally_valid(value) and len(re.sub(r"\D", "", value)) >= 9


def _entity_order(
    reconciliation: V21ReconciliationResult,
    spatial: V21SpatialRepresentation,
) -> dict[str, V21CanonicalEntity]:
    by_id = {entity.canonical_id: entity for entity in reconciliation.entities}
    return {
        entity_id: by_id[entity_id]
        for entity_id in spatial.read_order
        if entity_id in by_id
    }


def _same_row_sequences(
    reconciliation: V21ReconciliationResult,
    spatial: V21SpatialRepresentation,
) -> Iterable[Tuple[V21CanonicalEntity, ...]]:
    by_id = {entity.canonical_id: entity for entity in reconciliation.entities}
    for region in spatial.regions:
        for row in region.rows:
            entities = tuple(by_id[entity_id] for entity_id in row.entity_ids)
            for index in range(len(entities) - 1):
                yield entities[index : index + 2]


def _reconstructible_phone_pair(
    first: V21CanonicalEntity,
    second: V21CanonicalEntity,
    column_by_entity: dict[str, str],
) -> Optional[str]:
    first_text = _candidate_text(first)
    second_text = _candidate_text(second)
    if not _phone_fragment_text(first_text) or not _phone_fragment_text(second_text):
        return None
    if _phone_is_independent_complete(first_text) or _phone_is_independent_complete(
        second_text
    ):
        return None
    if column_by_entity.get(first.canonical_id) != column_by_entity.get(
        second.canonical_id
    ):
        return None

    first_box = first.canonical_bbox
    second_box = second.canonical_bbox
    first_width = max(first_box[2] - first_box[0], 0.0)
    second_width = max(second_box[2] - second_box[0], 0.0)
    horizontal_overlap = max(0.0, first_box[2] - second_box[0])
    smaller_width = max(min(first_width, second_width), 1.0)
    if second_box[0] < first_box[2] and (
        horizontal_overlap / smaller_width > 0.25
        or (second_box[0] + second_box[2]) / 2 <= (first_box[0] + first_box[2]) / 2
    ):
        return None
    first_height = max(first_box[3] - first_box[1], 0.0)
    second_height = max(second_box[3] - second_box[1], 0.0)
    local_height = max(first_height, second_height, 1.0)
    gap = second_box[0] - first_box[2]
    if gap > local_height * 1.5:
        return None
    vertical_intersection = max(
        0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    )
    if vertical_intersection / local_height < 0.5:
        return None

    combined = first_text + second_text
    if not _phone_is_structurally_valid(combined):
        return None
    digits = re.sub(r"\D", "", combined)
    if not 7 <= len(digits) <= 15:
        return None
    if (
        len(re.sub(r"\D", "", first_text)) < 3
        or len(re.sub(r"\D", "", second_text)) < 2
    ):
        return None
    if not re.search(r"[+()./\-]", first_text):
        return None
    return _normalize_phone(combined)


def extract_deterministic_candidates(
    reconciliation: V21ReconciliationResult,
    spatial: V21SpatialRepresentation,
) -> Tuple[V21ContactCandidate, ...]:
    """Extract non-semantic contact candidates from canonical spatial entities."""
    candidates: list[tuple[str, str, Tuple[str, ...]]] = []
    ordered_entities = _entity_order(reconciliation, spatial)
    for entity_id, entity in ordered_entities.items():
        text = _candidate_text(entity)
        for match in _EMAIL_CANDIDATE_RE.finditer(text):
            candidates.append(("EMAIL", _normalize_email(match.group()), (entity_id,)))
        email_spans = [match.span() for match in _EMAIL_CANDIDATE_RE.finditer(text)]
        for match in _SOCIAL_URL_CANDIDATE_RE.finditer(text):
            candidates.append(
                ("SOCIAL", _normalize_website(match.group()), (entity_id,))
            )
        for match in _WEBSITE_CANDIDATE_RE.finditer(text):
            if any(start <= match.start() < end for start, end in email_spans):
                continue
            if _SOCIAL_URL_CANDIDATE_RE.fullmatch(match.group()):
                continue
            candidates.append(
                ("WEBSITE", _normalize_website(match.group()), (entity_id,))
            )
        for match in _PHONE_CANDIDATE_RE.finditer(text):
            if _phone_is_structurally_valid(match.group()):
                candidates.append(
                    ("PHONE", _normalize_phone(match.group()), (entity_id,))
                )
        for match in _HANDLE_CANDIDATE_RE.finditer(text):
            if _EMAIL_CANDIDATE_RE.search(text):
                continue
            candidates.append(("SOCIAL", match.group().lower(), (entity_id,)))

    for region in spatial.regions:
        by_id = {entity.canonical_id: entity for entity in reconciliation.entities}
        column_by_entity = {
            entity_id: column_id
            for entity_id, column_id in spatial.entity_column_ids
            if entity_id in region.entity_ids
        }
        for row in region.rows:
            for first_id, second_id in zip(row.entity_ids, row.entity_ids[1:]):
                first = by_id[first_id]
                second = by_id[second_id]
                reconstructed = _reconstructible_phone_pair(
                    first,
                    second,
                    column_by_entity,
                )
                if reconstructed is None:
                    continue
                candidates.append(
                    (
                        "PHONE",
                        reconstructed,
                        (first.canonical_id, second.canonical_id),
                    )
                )

    result = []
    for index, (field_type, normalized_value, source_ids) in enumerate(candidates, 1):
        result.append(
            V21ContactCandidate(
                candidate_id=f"candidate-{index:06d}",
                field_type=field_type,
                normalized_value=normalized_value,
                source_canonical_entity_ids=source_ids,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class V21PanelPrimacyScore:
    region_id: str
    score: float
    rank: int
    identity_signal: float
    contact_signal: float
    density_signal: float
    entity_count: int
    contact_candidate_count: int


@dataclass(frozen=True)
class V21PanelPrimacyResult:
    ranked_regions: Tuple[V21PanelPrimacyScore, ...]
    primary_region_id: str


def _structural_identity_signal(
    entities: Iterable[V21CanonicalEntity],
    contact_entity_ids: set[str],
) -> float:
    entity_list = [
        entity for entity in entities if entity.canonical_id not in contact_entity_ids
    ]
    if not entity_list:
        return 0.0
    heights = sorted(_height(entity.canonical_bbox) for entity in entity_list)
    baseline = max(median(heights), 1.0)
    signals = []
    for entity in entity_list:
        text = _candidate_text(entity)
        words = [word for word in text.split() if word]
        non_space = [char for char in text if not char.isspace()]
        alpha_density = sum(char.isalpha() for char in non_space) / max(
            len(non_space), 1
        )
        digit_density = sum(char.isdigit() for char in non_space) / max(
            len(non_space), 1
        )
        compactness = min(1.0, 3.0 / max(len(words), 1))
        prominence = min(2.0, _height(entity.canonical_bbox) / baseline) / 2.0
        signals.append(prominence * compactness * alpha_density * (1.0 - digit_density))
    return max(signals, default=0.0)


def determine_panel_primacy(
    reconciliation: V21ReconciliationResult,
    spatial: V21SpatialRepresentation,
    candidates: Tuple[V21ContactCandidate, ...],
) -> V21PanelPrimacyResult:
    """Rank existing spatial regions using soft, non-semantic structural signals."""
    entities_by_id = {entity.canonical_id: entity for entity in reconciliation.entities}
    candidate_region_ids: dict[str, set[str]] = {}
    for region in spatial.regions:
        region_entities = set(region.entity_ids)
        candidate_region_ids[region.region_id] = {
            candidate.candidate_id
            for candidate in candidates
            if region_entities.intersection(candidate.source_canonical_entity_ids)
        }

    preliminary = []
    for region in spatial.regions:
        entities = tuple(entities_by_id[entity_id] for entity_id in region.entity_ids)
        contact_ids = {
            source_id
            for candidate in candidates
            if candidate.candidate_id in candidate_region_ids[region.region_id]
            for source_id in candidate.source_canonical_entity_ids
        }
        contact_count = len(candidate_region_ids[region.region_id])
        entity_count = len(entities)
        identity_signal = _structural_identity_signal(entities, contact_ids)
        type_count = len(
            {
                candidate.field_type
                for candidate in candidates
                if candidate.candidate_id in candidate_region_ids[region.region_id]
            }
        )
        contact_coverage = min(1.0, contact_count / max(entity_count, 1) * 3.0)
        contact_signal = 0.6 * contact_coverage + 0.4 * (type_count / 4.0)
        non_contact_count = max(entity_count - len(contact_ids), 0)
        density_signal = non_contact_count / max(entity_count, 1)
        score = 0.55 * identity_signal + 0.30 * contact_signal + 0.15 * density_signal
        preliminary.append(
            (
                region.region_id,
                score,
                identity_signal,
                contact_signal,
                density_signal,
                entity_count,
                contact_count,
            )
        )

    preliminary.sort(key=lambda item: (-item[1], item[0]))
    ranked = tuple(
        V21PanelPrimacyScore(
            region_id=region_id,
            score=round(score, 6),
            rank=index,
            identity_signal=round(identity_signal, 6),
            contact_signal=round(contact_signal, 6),
            density_signal=round(density_signal, 6),
            entity_count=entity_count,
            contact_candidate_count=contact_count,
        )
        for index, (
            region_id,
            score,
            identity_signal,
            contact_signal,
            density_signal,
            entity_count,
            contact_count,
        ) in enumerate(preliminary, 1)
    )
    return V21PanelPrimacyResult(
        ranked_regions=ranked,
        primary_region_id=ranked[0].region_id if ranked else "",
    )


# Phase 5 bounds are deliberately separate: limiting the final result does not
# permit unbounded edge or cut-set enumeration.
V21_MAX_OUTLIER_EDGES = 12
V21_MAX_CUT_SETS = 12
V21_MAX_HYPOTHESES_PER_REGION = 12
V21_MAX_TOTAL_HYPOTHESES = 64


@dataclass(frozen=True)
class V21HypothesisGroup:
    group_id: str
    entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class V21SubregionHypothesis:
    hypothesis_id: str
    region_id: str
    groups: Tuple[V21HypothesisGroup, ...]
    score: float
    rank: int
    cut_edge_ids: Tuple[Tuple[str, str], ...] = ()
    score_components: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class V21SubregionRegionDiagnostics:
    region_id: str
    candidate_outlier_edges_considered: int
    candidate_cut_sets_evaluated: int
    hypotheses_returned: int


@dataclass(frozen=True)
class V21SubregionHypothesisResult:
    hypotheses: Tuple[V21SubregionHypothesis, ...]
    diagnostics: Tuple[V21SubregionRegionDiagnostics, ...] = ()


def _phase5_components(
    entity_ids: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    removed: frozenset[Tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    adjacency = {entity_id: set() for entity_id in entity_ids}
    for first, second in edges:
        edge = tuple(sorted((first, second)))
        if edge in removed:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = []
    visited = set()
    for entity_id in sorted(adjacency):
        if entity_id in visited:
            continue
        pending = [entity_id]
        component = []
        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            pending.extend(sorted(adjacency[current] - visited))
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _phase5_intervening_count(
    edge: V21LinkedPairEvidence,
    edges: tuple[V21LinkedPairEvidence, ...],
) -> int:
    scale = max(edge.scale, 1.0)
    first_center = _center(edge.first_bbox)
    second_center = _center(edge.second_bbox)
    dx = second_center[0] - first_center[0]
    dy = second_center[1] - first_center[1]
    length = hypot(dx, dy)
    if length <= 0.0:
        return 0
    intervening = 0
    for other in edges:
        if other is edge:
            continue
        for candidate_center in (_center(other.first_bbox), _center(other.second_bbox)):
            projection = (
                (candidate_center[0] - first_center[0]) * dx
                + (candidate_center[1] - first_center[1]) * dy
            ) / length
            perpendicular = (
                abs(
                    (candidate_center[0] - first_center[0]) * dy
                    - (candidate_center[1] - first_center[1]) * dx
                )
                / length
            )
            if 0.0 < projection < length and perpendicular <= scale:
                intervening += 1
                break
    return intervening


def _phase5_edge_strength(
    edge: V21LinkedPairEvidence,
    edges: tuple[V21LinkedPairEvidence, ...],
) -> float:
    scale = max(edge.scale, 1.0)
    edge_gap = (
        max(0.0, -edge.horizontal_overlap) + max(0.0, -edge.vertical_overlap)
    ) / scale
    competing = []
    for other in edges:
        if other is edge:
            continue
        if edge.first_entity_id in (other.first_entity_id, other.second_entity_id) or (
            edge.second_entity_id in (other.first_entity_id, other.second_entity_id)
        ):
            competing.append(
                (
                    max(0.0, -other.horizontal_overlap)
                    + max(0.0, -other.vertical_overlap)
                )
                / max(other.scale, 1.0)
            )
    if not competing:
        return 0.0
    local_gap = min(competing)
    anomaly = max(0.0, edge_gap - local_gap)
    return anomaly / (1.0 + _phase5_intervening_count(edge, edges))


def _phase5_cut_sets(
    edges: tuple[V21LinkedPairEvidence, ...],
) -> tuple[tuple[Tuple[str, str], ...], ...]:
    if not edges:
        return ()
    seeds = [
        edge
        for edge in edges
        if _phase5_edge_strength(edge, edges) > 0.0
        and _phase5_intervening_count(edge, edges) == 0
    ]
    if not seeds:
        return ()
    seed_ids = {
        entity_id
        for edge in seeds
        for entity_id in (edge.first_entity_id, edge.second_entity_id)
    }
    related = [
        edge
        for edge in edges
        if edge not in seeds
        and (edge.first_entity_id in seed_ids or edge.second_entity_id in seed_ids)
    ]
    ranked = sorted(
        (*seeds, *related),
        key=lambda edge: (
            -_phase5_edge_strength(edge, edges),
            _phase5_intervening_count(edge, edges),
            edge.first_entity_id,
            edge.second_entity_id,
        ),
    )[:V21_MAX_OUTLIER_EDGES]
    edge_ids = tuple((edge.first_entity_id, edge.second_entity_id) for edge in ranked)
    cut_sets: list[tuple[Tuple[str, str], ...]] = [
        tuple(sorted(edge_ids[:size])) for size in range(1, min(3, len(edge_ids)) + 1)
    ]
    for size in range(1, min(3, len(edge_ids)) + 1):
        for candidate in combinations(edge_ids, size):
            normalized = tuple(sorted(candidate))
            if normalized in cut_sets:
                continue
            cut_sets.append(normalized)
            if len(cut_sets) >= V21_MAX_CUT_SETS:
                return tuple(cut_sets)
    return tuple(cut_sets)


def _phase5_score(
    groups: tuple[tuple[str, ...], ...],
    entity_by_id: dict[str, V21CanonicalEntity],
    cut_edges: tuple[Tuple[str, str], ...],
    all_edges: tuple[V21LinkedPairEvidence, ...],
    candidates: Tuple[V21ContactCandidate, ...],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    boxes = [
        entity_by_id[entity_id].canonical_bbox
        for group in groups
        for entity_id in group
    ]
    widths = [_width(box) for box in boxes]
    heights = [_height(box) for box in boxes]
    scale = max(median(widths + heights), 1.0)
    compactness_values = []
    for group in groups:
        group_boxes = [entity_by_id[item].canonical_bbox for item in group]
        envelope = (
            min(box[0] for box in group_boxes),
            min(box[1] for box in group_boxes),
            max(box[2] for box in group_boxes),
            max(box[3] for box in group_boxes),
        )
        envelope_area = max(_width(envelope) * _height(envelope), scale * scale)
        member_area = sum(_width(box) * _height(box) for box in group_boxes)
        compactness_values.append(min(1.0, member_area / envelope_area))
    coherence = sum(compactness_values) / max(len(compactness_values), 1)
    separation_edges = {tuple(sorted(edge)) for edge in cut_edges}
    separation = sum(
        (edge.normalized_dx + edge.normalized_dy) / 2.0
        for edge in all_edges
        if (edge.first_entity_id, edge.second_entity_id) in separation_edges
    ) / max(len(cut_edges), 1)
    separation = min(1.0, separation / 3.0)
    balance = min(len(group) for group in groups) / max(len(group) for group in groups)
    anchor_ids = {
        entity_id
        for candidate in candidates
        for entity_id in candidate.source_canonical_entity_ids
    }
    anchor_concentration = (
        1.0 if any(anchor_ids.intersection(group) for group in groups) else 0.0
    )
    score = (
        0.35 * coherence
        + 0.30 * separation
        + 0.20 * balance
        + 0.15 * anchor_concentration
    )
    components = (
        ("coherence", round(coherence, 6)),
        ("separation", round(separation, 6)),
        ("balance", round(balance, 6)),
        ("anchor_concentration", round(anchor_concentration, 6)),
    )
    return round(score, 6), components


def generate_subregion_hypotheses(
    reconciliation: V21ReconciliationResult,
    spatial: V21SpatialRepresentation,
    evidence: V21SpatialEvidence,
    candidates: Tuple[V21ContactCandidate, ...],
) -> V21SubregionHypothesisResult:
    """Generate bounded, crisp spatial alternatives inside existing regions.

    H0 is always emitted. Split candidates remove deterministic subsets of at
    most three strongest normalized outlier edges. A singleton is rejected
    unless it has at least two independent linked neighbours in the induced
    graph, preventing one weak peripheral edge from manufacturing a panel.
    """
    entities = {entity.canonical_id: entity for entity in reconciliation.entities}
    evidence_by_region = {
        region.region_id: tuple(
            sorted(
                (
                    item
                    for item in evidence.linked_pair_evidence
                    if item.first_entity_id in region.entity_ids
                    and item.second_entity_id in region.entity_ids
                ),
                key=lambda item: (item.first_entity_id, item.second_entity_id),
            )
        )
        for region in spatial.regions
    }
    all_hypotheses = []
    diagnostics = []
    for region in spatial.regions:
        region_ids = tuple(sorted(region.entity_ids))
        region_edges = evidence_by_region[region.region_id]
        edge_ids = tuple(
            (item.first_entity_id, item.second_entity_id) for item in region_edges
        )
        h0_groups = (
            V21HypothesisGroup(f"{region.region_id}-group-000001", region_ids),
        )
        h0_score, h0_components = _phase5_score(
            (region_ids,), entities, (), region_edges, candidates
        )
        region_hypotheses = [
            V21SubregionHypothesis(
                hypothesis_id=f"{region.region_id}-hypothesis-000001",
                region_id=region.region_id,
                groups=h0_groups,
                score=h0_score,
                rank=0,
                score_components=h0_components,
            )
        ]
        cut_sets = _phase5_cut_sets(region_edges) if len(region_ids) > 2 else ()
        accepted = []
        neighbours = {entity_id: set() for entity_id in region_ids}
        for first, second in edge_ids:
            neighbours[first].add(second)
            neighbours[second].add(first)
        for cut_set in cut_sets:
            groups = _phase5_components(region_ids, edge_ids, frozenset(cut_set))
            if len(groups) <= 1:
                continue
            if any(
                len(group) == 1 and len(neighbours[group[0]]) < 2 for group in groups
            ):
                continue
            score, components = _phase5_score(
                groups, entities, cut_set, region_edges, candidates
            )
            accepted.append(
                V21SubregionHypothesis(
                    hypothesis_id="",
                    region_id=region.region_id,
                    groups=tuple(
                        V21HypothesisGroup(
                            f"{region.region_id}-group-{index:06d}", group
                        )
                        for index, group in enumerate(groups, 1)
                    ),
                    score=score,
                    rank=0,
                    cut_edge_ids=cut_set,
                    score_components=components,
                )
            )
        accepted.sort(
            key=lambda item: (
                -item.score,
                len(item.groups),
                tuple(
                    entity_id for group in item.groups for entity_id in group.entity_ids
                ),
                item.cut_edge_ids,
            )
        )
        region_hypotheses.extend(accepted[: V21_MAX_HYPOTHESES_PER_REGION - 1])
        ranked = []
        for index, item in enumerate(
            sorted(
                region_hypotheses,
                key=lambda candidate: (
                    -candidate.score,
                    len(candidate.groups),
                    tuple(
                        entity_id
                        for group in candidate.groups
                        for entity_id in group.entity_ids
                    ),
                    candidate.cut_edge_ids,
                ),
            ),
            1,
        ):
            ranked.append(
                V21SubregionHypothesis(
                    **{
                        **item.__dict__,
                        "hypothesis_id": f"{region.region_id}-hypothesis-{index:06d}",
                        "rank": index,
                    }
                )
            )
        all_hypotheses.extend(ranked)
        diagnostics.append(
            V21SubregionRegionDiagnostics(
                region_id=region.region_id,
                candidate_outlier_edges_considered=min(
                    len(region_edges), V21_MAX_OUTLIER_EDGES
                ),
                candidate_cut_sets_evaluated=min(len(cut_sets), V21_MAX_CUT_SETS),
                hypotheses_returned=len(ranked),
            )
        )
    all_hypotheses = all_hypotheses[:V21_MAX_TOTAL_HYPOTHESES]
    return V21SubregionHypothesisResult(
        hypotheses=tuple(all_hypotheses), diagnostics=tuple(diagnostics)
    )


def _result_mapping(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("res", result)
    value = getattr(result, "json", None)
    if isinstance(value, dict):
        return value.get("res", value)
    return result


def _raw_observations(results: Any) -> Iterable[tuple[str, float, list[list[float]]]]:
    """Read Paddle output without stripping, normalizing, filtering, or merging."""
    if not results:
        return

    for result in results:
        mapped = _result_mapping(result)
        if isinstance(mapped, dict):
            texts = mapped.get("rec_texts", []) or []
            scores = mapped.get("rec_scores", []) or []
            polygons = mapped.get("rec_polys", mapped.get("dt_polys", [])) or []
            for text, confidence, polygon in zip(texts, scores, polygons):
                points = (
                    polygon.tolist() if isinstance(polygon, np.ndarray) else polygon
                )
                yield text, float(confidence), [
                    [float(point[0]), float(point[1])] for point in points
                ]
            continue

        try:
            items = list(mapped)
        except TypeError:
            continue
        for item in items:
            if len(item) < 2:
                continue
            text_confidence = item[1]
            if (
                not isinstance(text_confidence, (list, tuple))
                or len(text_confidence) < 2
            ):
                continue
            polygon = item[0]
            yield text_confidence[0], float(text_confidence[1]), [
                [float(point[0]), float(point[1])] for point in polygon
            ]


def _capture_pass(
    image: np.ndarray, lang: str, pass_id: str, pool: V21EvidencePool
) -> int:
    ocr = _get_paddle(lang)
    added = 0
    for raw_text, confidence, bbox in _raw_observations(ocr.predict(image)):
        pool.append(
            pass_id=pass_id,
            raw_text=raw_text,
            bbox=bbox,
            confidence=confidence,
            metadata={"language": lang},
        )
        added += 1
    return added


def process_image(
    image: np.ndarray, pool_id: str = "card", lang: str = "en"
) -> V21EvidencePool:
    processed = preprocess(deskew(image))
    pool = V21EvidencePool(pool_id=pool_id)
    _capture_pass(processed, lang, "pass_1", pool)
    enhanced = cv2.convertScaleAbs(
        processed,
        alpha=CFG["OCR_BRIGHTNESS_ALPHA"],
        beta=CFG["OCR_BRIGHTNESS_BETA"],
    )
    _capture_pass(enhanced, lang, "pass_2", pool)
    return pool


def build_pipeline_representation(
    pool: V21EvidencePool,
) -> tuple[V21ReconciliationResult, V21SpatialRepresentation]:
    """Build derived Phase 2 and Phase 3 views without changing raw evidence."""
    reconciled = reconcile_evidence(pool)
    spatial = build_spatial_representation(reconciled)
    return reconciled, spatial


def _images(target: Path) -> Iterable[tuple[str, np.ndarray]]:
    paths = sorted(target.iterdir()) if target.is_dir() else [target]
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("Could not read image: %s", path)
            continue
        yield path.name, image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visiting Card OCR Engine V21")
    parser.add_argument("target", type=Path)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for label, image in _images(args.target):
        pool = process_image(image, pool_id=Path(label).stem, lang=args.lang)
        reconciled, spatial = build_pipeline_representation(pool)
        candidates = extract_deterministic_candidates(reconciled, spatial)
        primacy = determine_panel_primacy(reconciled, spatial, candidates)
        if args.as_json:
            print(
                json.dumps(
                    {"source": label, "evidence": pool.to_dict()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue
        pass_1_count = len(pool.by_pass("pass_1"))
        pass_2_count = len(pool.by_pass("pass_2"))
        row_count = sum(len(region.rows) for region in spatial.regions)
        column_count = sum(len(region.columns) for region in spatial.regions)
        print(f"{label}")
        print(
            f"  raw observations={len(pool.observations)} "
            f"pass_1={pass_1_count} pass_2={pass_2_count} "
            f"canonical entities={len(reconciled.entities)} "
            f"regions={len(spatial.regions)} rows={row_count} columns={column_count}"
        )
        print("  spatial read order:")
        for region in spatial.regions:
            print(f"    {region.region_id} bbox={region.bbox}")
            for row in region.rows:
                print(f"      {row.row_id}")
                for entity_id, column_id in zip(row.entity_ids, row.column_ids):
                    entity = next(
                        item
                        for item in reconciled.entities
                        if item.canonical_id == entity_id
                    )
                    print(
                        f"        {entity_id} column={column_id} "
                        f"text={entity.representative_text!r}"
                    )
        print("  deterministic candidates:")
        for field_type in ("EMAIL", "WEBSITE", "PHONE", "SOCIAL"):
            field_candidates = [
                candidate
                for candidate in candidates
                if candidate.field_type == field_type
            ]
            if not field_candidates:
                continue
            print(f"    {field_type}")
            for candidate in field_candidates:
                print(f"      {candidate.normalized_value}")
                print(
                    "        sources: "
                    + ", ".join(candidate.source_canonical_entity_ids)
                )
        print("  panel primacy:")
        for panel in primacy.ranked_regions:
            print(
                f"    rank={panel.rank} {panel.region_id} "
                f"score={panel.score:.6f} "
                f"identity={panel.identity_signal:.6f} "
                f"contact={panel.contact_signal:.6f}"
            )
        print(f"    primary={primacy.primary_region_id}")


if __name__ == "__main__":
    main()
