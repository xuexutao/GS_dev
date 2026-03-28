#!/usr/bin/env python3
"""
Wrapper script to run semantic label assignment with reduced output
"""
import os
import sys
import tqdm

# Disable tqdm output
tqdm.tqdm = lambda iterable, desc=None, **kwargs: iterable

# Now import and run the main script
from assign_semantic_to_trained_model import main

if __name__ == "__main__":
    main()