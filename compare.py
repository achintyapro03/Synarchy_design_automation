import xml.etree.ElementTree as ET
import sys
import re
from collections import defaultdict
from typing import Dict, Set

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def strip_comments(xml_text: str) -> str:
    """Remove XML comments (<!-- ... -->) using regex."""
    return re.sub(r'<!--.*?-->', '', xml_text, flags=re.DOTALL)

def collect_params(elem: ET.Element, prefix: str = "", 
                   params: Dict[str, Set[str]] = None) -> Dict[str, Set[str]]:
    """Recursively collect param names keyed by full component path."""
    if params is None:
        params = defaultdict(set)

    comp_id = elem.attrib.get('id', elem.attrib.get('name', 'unknown'))
    path = f"{prefix}.{comp_id}" if prefix else comp_id

    for param in elem.findall('param'):
        params[path].add(param.attrib['name'])

    for child in elem.findall('component'):
        collect_params(child, path, params)

    return params

def compare_dicts(dict1: Dict[str, Set[str]], dict2: Dict[str, Set[str]], 
                  label1: str, label2: str) -> str:
    """Compare two dictionaries of sets and return colorized differences."""
    diffs = []
    all_paths = set(dict1.keys()) | set(dict2.keys())
    for path in sorted(all_paths):
        set1 = dict1.get(path, set())
        set2 = dict2.get(path, set())
        only1 = sorted(set1 - set2)
        only2 = sorted(set2 - set1)
        if only1 or only2:
            diffs.append(f"\n=== Component: {path} ===")
            if only1:
                diffs.append(f"  {RED}Present in {label1} but not in {label2}: {', '.join(only1)}{RESET}")
            if only2:
                diffs.append(f"  {GREEN}Present in {label2} but not in {label1}: {', '.join(only2)}{RESET}")
    return "\n".join(diffs)

def parse_file(filename: str) -> ET.Element:
    """Read XML file and strip comments."""
    with open(filename, 'r') as f:
        xml_text = f.read()
    clean_xml = strip_comments(xml_text)
    return ET.fromstring(clean_xml)

def main(file1: str, file2: str) -> None:
    root1 = parse_file(file1)
    root2 = parse_file(file2)

    params1 = collect_params(root1)
    params2 = collect_params(root2)

    print("### Parameter Differences ###")
    print(compare_dicts(params1, params2, file1, file2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_mcpat_xml.py <file1.xml> <file2.xml>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
