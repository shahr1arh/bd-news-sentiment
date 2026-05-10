"""
download_nltk.py — Pre-download all NLTK data packages needed by this project.

Run this ONCE during your build or setup step so the server never has to
download data at runtime (which fails on ephemeral cloud filesystems).

Usage:
    python download_nltk.py

Cloud deployment (add to your build command on Render / Railway):
    pip install -r requirements.txt && python download_nltk.py
"""

import sys
import nltk

PACKAGES = [
    "vader_lexicon",   # Required by analyzer.py for sentiment scoring
]

def main():
    all_ok = True
    for pkg in PACKAGES:
        print(f"  Downloading '{pkg}' ...", end=" ", flush=True)
        try:
            nltk.download(pkg, quiet=False, raise_on_error=True)
            print("✓")
        except Exception as exc:
            print(f"✗  ERROR: {exc}")
            all_ok = False

    if all_ok:
        print("\nAll NLTK packages downloaded successfully.")
    else:
        print("\nSome packages failed to download. Check your network connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
