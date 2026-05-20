from gpqa_cmab.subsets import all_subsets, subset_id


def test_all_subsets_are_stable():
    assert [subset_id(item) for item in all_subsets()] == [
        "main_only",
        "A",
        "B",
        "C",
        "D",
        "A,B",
        "A,C",
        "A,D",
        "B,C",
        "B,D",
        "C,D",
        "A,B,C",
        "A,B,D",
        "A,C,D",
        "B,C,D",
        "A,B,C,D",
    ]


def test_subset_id_sorts_agents():
    assert subset_id({"D", "A"}) == "A,D"
