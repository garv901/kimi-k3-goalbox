"""Negative control: build() raises."""


def build(case: dict):
    raise ImportError("deliberate failure in build")
