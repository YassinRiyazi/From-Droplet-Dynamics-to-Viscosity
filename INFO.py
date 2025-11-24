import os
import re

def count_python_lines(root_dir: str):
    """
    Count total and code-only lines (excluding comments and docstrings)
    for all Python files under the given directory.

    Args:
        root_dir (str): Path to start searching for .py files.

    Returns:
        tuple[int, int, int]: (file_count, total_lines, code_lines)
    """
    total_lines = 0
    code_lines = 0
    file_count = 0

    triple_quote_pattern = re.compile(r'("""|\'\'\')')

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'BK' in dirpath:
                continue  # Skip .BK directories

            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        in_docstring = False
                        file_total = 0
                        file_code = 0

                        for line in f:
                            file_total += 1
                            stripped = line.strip()

                            # Skip empty lines
                            if not stripped:
                                continue

                            # Handle docstrings
                            if triple_quote_pattern.search(stripped):
                                quote_count = stripped.count('"""') + stripped.count("'''")
                                if quote_count % 2 != 0:
                                    in_docstring = not in_docstring
                                continue

                            if in_docstring:
                                continue

                            # Skip comments and shebangs
                            if stripped.startswith("#") or stripped.startswith("#!"):
                                continue

                            file_code += 1

                        print(f"{file_path}: {file_total} total lines, {file_code} code lines")

                        total_lines += file_total
                        code_lines += file_code
                        file_count += 1

                except Exception as e:
                    print(f"⚠️ Could not read {file_path}: {e}")

    print(f"\n📊 Total Python files: {file_count}")
    print(f"📏 Total lines (including comments & docstring): {total_lines}")
    print(f"💻 Total code lines (excluding comments & docstring): {code_lines}")
    return file_count, total_lines, code_lines


if __name__ == "__main__":
    # Change "." to the path of your project root
    project_root = "."
    count_python_lines(project_root)

