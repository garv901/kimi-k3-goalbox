"""Negative control: the candidate never returns (deadline test)."""
import time


def build(case: dict):
    def run(inputs: dict) -> dict:
        time.sleep(3600)
        return {}

    return run
