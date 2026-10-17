import pytest

from ml.datasets.build_pest_dataset import (
    CONFIRMED_CLASSES, REVIEW_FILE, SOURCES, build_id_map, original_stem, parse_label_line, remap_labels,
)


def test_original_stem_strips_roboflow_suffix():
    assert original_stem("IMG_12_jpg.rf.ab12cd34.jpg") == "IMG_12"
    assert original_stem("leaf-3_png.rf.0f.jpg") == "leaf-3"
    assert original_stem("plain_name.jpg") == "plain_name"


@pytest.mark.parametrize("line,ok", [
    ("0 0.5 0.5 0.1 0.1", True),
    ("2 0 1 1 1", True),
    ("0 0.5 0.5 0.1", False),        # wrong arity
    ("a 0.5 0.5 0.1 0.1", False),    # non-numeric class
    ("0 1.2 0.5 0.1 0.1", False),    # out of range
    ("0 0.5 0.5 0 0.1", False),      # empty box
    ("-1 0.5 0.5 0.1 0.1", False),   # negative class
])
def test_parse_label_line_validates(line, ok):
    assert (parse_label_line(line) is not None) == ok


def test_remap_labels_keeps_only_mapped_classes():
    lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "garbage", ""]
    out, stats = remap_labels(lines, {0: 2})
    assert out == ["2 0.500000 0.500000 0.100000 0.100000"]
    assert stats == {"kept": 1, "dropped_other_class": 1, "invalid": 1}


def test_build_id_map_maps_to_confirmed_index_and_skips_gated_names():
    id_map = build_id_map(["mealybug", "mango hopper", "other"], "Mealy Bug", None)
    assert id_map == {0: CONFIRMED_CLASSES.index("Mealy Bug"), 2: CONFIRMED_CLASSES.index("Mealy Bug")}
    assert build_id_map(["fly", "bee"], "Fruit Fly", ["fly"]) == {0: 0}


@pytest.mark.parametrize("target", ["Mango Hopper", "Stem Borer", "Aphid"])
def test_build_id_map_refuses_gated_or_unknown_targets(target):
    with pytest.raises(ValueError):
        build_id_map(["x"], target, None)


def test_parse_label_line_converts_polygon_to_enclosing_box():
    cls, x, y, w, h = parse_label_line("1 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7")
    assert cls == 1
    assert (x, y, w, h) == pytest.approx((0.4, 0.5, 0.4, 0.4))


@pytest.mark.parametrize("line", ["1 0.2 0.3 0.6 0.3 0.6 0.4 0.5", "1 0.2 0.3 0.6 0.3 0.6", "1 0.2 0.2 0.2 0.2 0.2 0.2"])
def test_parse_label_line_rejects_degenerate_polygons(line):
    assert parse_label_line(line) is None


def test_every_source_maps_only_named_source_classes():
    # None would map every source class (e.g. 'Grape', 'Coffee Leaf Rust', 'Pest') onto the pest target
    for s in SOURCES:
        assert s["source_classes"], s["slug"]
        assert s["target"] in CONFIRMED_CLASSES


def test_review_file_records_a_host_for_every_source():
    import json
    review = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    for s in SOURCES:
        host = review[s["slug"]]["host"]
        assert host == "mango" or host == "unknown" or host.startswith("other:")
