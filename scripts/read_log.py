import sys
from pathlib import Path


def read_log(path):
    encodings = ["utf-16", "utf-8", "cp1252", "latin1"]
    content = None
    for enc in encodings:
        try:
            content = Path(path).read_text(encoding=enc)
            print(f"Read successfully with encoding: {enc}")
            break
        except Exception:
            continue

    if content:
        print("-" * 20)
        print(content)
        print("-" * 20)

        # Check for success indicators
        if "Enriching with External Data" in content:
            print("SUCCESS: Enrichment logic started.")
        else:
            print("WARNING: 'Enriching with External Data' NOT found.")

        if "External features added" in content:
            print("SUCCESS: External features added to dataframe.")
        else:
            print("WARNING: 'External features added' NOT found.")

    else:
        print("Failed to read file with any encoding.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_log(sys.argv[1])
    else:
        print("Usage: python read_log.py <path>")
