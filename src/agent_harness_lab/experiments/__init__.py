"""Experiment definitions and benchmark-matrix orchestration (``ahl benchmark``).

An experiment file says WHAT to benchmark; the existing run-plan, executor and evaluator layers
decide HOW each run and evaluation works. Nothing here talks to Docker directly or knows how a
harness is invoked.
"""
