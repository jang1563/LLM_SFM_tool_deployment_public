from post_training.summarize_sft_pressure_runs import (
    merge_class_accuracy,
    parse_fraction,
    rounded_range,
)


def test_parse_fraction_reads_class_accuracy_string() -> None:
    assert parse_fraction("3/8") == (3, 8)


def test_merge_class_accuracy_adds_numerators_and_denominators() -> None:
    merged = merge_class_accuracy([
        {"defer": "1/2", "flag": "0/2"},
        {"defer": "2/2", "flag": "1/2"},
    ])

    assert merged == {"defer": "3/4", "flag": "1/4"}


def test_rounded_range_formats_min_and_max() -> None:
    assert rounded_range([0.4, 0.7, 0.3]) == "0.300..0.700"
