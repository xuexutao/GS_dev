#!/usr/bin/env python3
"""
Run semantic label assignment with suppressed output
"""
import os
import sys
import contextlib
import io

# Redirect stdout to a buffer, but allow errors
class TeeOutput:
    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer
    
    def write(self, text):
        self.buffer.write(text)
        # Only print lines that contain error keywords or important info
        if any(keyword in text.lower() for keyword in ['error', 'warning', 'debug', 'loading semantic', 'saved', 'assigned']):
            self.original.write(text)
    
    def flush(self):
        self.original.flush()
        self.buffer.flush()

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--semantic_labels", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mask_dirname", default="masks_sam")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    # Import after parsing to avoid loading heavy modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Redirect stdout
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, buffer)
    
    try:
        from assign_semantic_to_trained_model import main as assign_main
        assign_main()
    finally:
        sys.stdout = original_stdout
        # Print summary from buffer
        buffer.seek(0)
        lines = buffer.readlines()
        # Filter out "Reading camera" lines
        for line in lines:
            if 'Reading camera' not in line and 'Loading cameras' not in line:
                original_stdout.write(line)
        original_stdout.flush()

if __name__ == "__main__":
    main()