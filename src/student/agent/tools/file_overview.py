import os


def get_file_message(root, max_depth=3):
    """Return a formatted overview of files/folders up to depth 3.

    Produces a readable tree (directories first) excluding ignored paths.
    """

    if not os.path.exists(root):
        return "\n\n<file_overview>\nTree:\n(NOT FOUND)\n</file_overview>\n"

    def list_children(base_path: str, base_root: str, current_depth: int):
        lines = []
        try:
            entries = sorted(os.listdir(base_path))
        except Exception:
            return lines

        # separate dirs and files; skip ignored
        dirs = []
        files = []
        for name in entries:
            full = os.path.join(base_path, name)
            rel = os.path.relpath(full, start=base_root)
            if check_ignore(rel):
                continue
            if os.path.isdir(full):
                dirs.append((name, full, rel))
            else:
                files.append((name, full, rel))

        # list directories first
        for name, full, rel in dirs:
            item_depth = current_depth + 1
            if item_depth <= max_depth:
                indent = "  " * (item_depth - 1)
                lines.append(f"{indent}- {name}/")
                # only descend if we haven't reached max depth
                if item_depth < max_depth:
                    lines.extend(list_children(full, base_root, item_depth))

        # then files
        for name, full, rel in files:
            item_depth = current_depth + 1
            if item_depth <= max_depth:
                indent = "  " * (item_depth - 1)
                lines.append(f"{indent}- {name}")

        return lines

    tree_lines = list_children(root, root, 0)
    tree_formatted = "\n".join(tree_lines) if tree_lines else "(empty)"

    # Nicely formatted overview block
    overview = (
        f"\n\n<file_overview>\n" f"Tree:\n{tree_formatted}\n" f"</file_overview>\n"
    )
    return overview


def check_ignore(file_name):
    # Return True is file should be ignored
    blacklist = [
        "Movies/",
        "VTK/",
        "Restart/",
        "run.sh",
        ".DS_Store",
        ".md",
        ".json",
        ".jsonl",
        ".log",
    ]
    for p in blacklist:
        if p in file_name:
            return True
    return False
