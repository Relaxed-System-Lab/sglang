"""Helpers for stage-local queue ordering in disaggregated serving."""


def sort_waiting_queue_by_policy(waiting_queue, policy, running_batch):
    """Sort a local waiting queue using the scheduler policy when available."""
    if policy is None or len(waiting_queue) <= 1:
        return
    policy.calc_priority(waiting_queue, running_batch)
