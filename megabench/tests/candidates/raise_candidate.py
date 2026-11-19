"""Negative control: the candidate raises inside run()."""


def build(case: dict):
    def run(inputs: dict) -> dict:
        raise RuntimeError("deliberate failure inside the candidate")

    return run
