from weighttraits.taskdata.assignment import assign_task_data


TASKS = {
    "alpha": {"datasets": [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}, {"id": "a4"}]},
    "beta": {"datasets": [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}, {"id": "b4"}]},
}


def test_assignment_is_deterministic_for_seed():
    rows = [
        {"node_id": "n0", "depth": 1, "grow": "train"},
        {"node_id": "n1", "depth": 2, "grow": "train"},
        {"node_id": "n2", "depth": 2, "grow": "train"},
    ]

    a = assign_task_data(rows, TASKS, seed=7)
    b = assign_task_data(rows, TASKS, seed=7)

    assert a == b
    assert all("task_family" in row and "dataset_id" in row for row in a)


def test_per_depth_policy_reuses_assignment_within_depth():
    rows = [
        {"node_id": "n0", "depth": 1, "grow": "train"},
        {"node_id": "n1", "depth": 2, "grow": "train"},
        {"node_id": "n2", "depth": 2, "grow": "train"},
    ]

    assigned = assign_task_data(rows, TASKS, seed=3, policy="per_depth")

    depth2 = [(row["task_family"], row["dataset_id"]) for row in assigned if row["depth"] == 2]
    assert len(set(depth2)) == 1


def test_can_restrict_task_families():
    rows = [{"node_id": "n0", "depth": 1, "grow": "train"}]

    assigned = assign_task_data(rows, TASKS, seed=1, task_families=["beta"])

    assert assigned[0]["task_family"] == "beta"
    assert assigned[0]["dataset_id"].startswith("b")

