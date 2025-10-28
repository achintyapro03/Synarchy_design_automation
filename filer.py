#!/usr/bin/env python3
import sys

def filter_lines_by_keywords(input_file: str, output_file: str, keywords: list[str]):
    """
    Keep only lines from input_file that contain any of the given keywords
    (case-insensitive substring match), and write them to output_file.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if any(k.lower() in line.lower() for k in keywords):
                outfile.write(line)

def main():
    """
    Usage:
        python3 filter_lines.py <input_file> <output_file> <keyword1> [keyword2 ...]
    Example:
        python3 filter_lines.py input.txt output.txt port cache miss
    """
    if len(sys.argv) < 4:
        print("Usage: python3 filter_lines.py <input_file> <output_file> <keyword1> [keyword2 ...]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    keywords = sys.argv[3:]

    filter_lines_by_keywords(input_file, output_file, keywords)
    print(f"[INFO] Filtered lines containing {keywords} written to {output_file}")

if __name__ == "__main__":
    main()
