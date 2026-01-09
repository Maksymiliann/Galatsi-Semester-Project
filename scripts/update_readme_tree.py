from __future__ import annotations
from pathlib import Path

START = "<!-- TREE_START -->"
END = "<!-- TREE_END -->"

# Default depth for everything
DEFAULT_DEPTH = 1

# Custom depth rules (relative to repo root)
# Use folder names exactly as they are in your repo (case + spaces!)
DEPTH_OVERRIDES = {
    "Src": 1,            # show more files in Src
    "Main code": 2,      # keep it readable
    "Results": 2,        # Results is huge -> stay shallow
    "runs": 1,           # but...
    "Dataset": 2,
    #"runs/obb": 2,       # ...limit obb a bit (optional)
}

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode"}
IGNORE_FILES = {".DS_Store"}

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def should_ignore(path: Path) -> bool:
    name = path.name
    if path.is_dir() and name in IGNORE_DIRS:
        return True
    if path.is_file() and name in IGNORE_FILES:
        return True
    return False


def get_allowed_depth(relative_path: str) -> int:
    """
    Return max depth allowed under a given relative path (folder).
    If no override exists, return DEFAULT_DEPTH.
    Longest matching prefix wins.
    """
    best = None
    best_len = -1
    for key, depth in DEPTH_OVERRIDES.items():
        if relative_path == key or relative_path.startswith(key + "/"):
            if len(key) > best_len:
                best = depth
                best_len = len(key)
    return best if best is not None else DEFAULT_DEPTH


def build_tree(root: Path) -> str:
    lines: list[str] = [f"{root.name}/"]

    def walk(dir_path: Path, prefix: str, depth_from_root: int):
        # Determine allowed depth for this subtree
        rel = dir_path.relative_to(root).as_posix()
        allowed = get_allowed_depth(rel) if rel != "." else DEFAULT_DEPTH

        if depth_from_root >= allowed:
            return

        entries = [p for p in dir_path.iterdir() if not should_ignore(p)]
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))  # dirs first then files

        for i, p in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + (p.name + ("/" if p.is_dir() else "")))

            if p.is_dir():
                extension = "    " if is_last else "│   "
                walk(p, prefix + extension, depth_from_root + 1)

    walk(root, "", 0)
    return "\n".join(lines)


def inject_into_readme(readme_path: Path, tree_text: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    if START not in content or END not in content:
        raise RuntimeError(f"README.md must contain {START} and {END} markers.")

    before = content.split(START)[0]
    after = content.split(END)[1]

    injected = (
        before
        + START
        + "\n\n```text\n"
        + tree_text
        + "\n```\n\n"
        + END
        + after
    )

    readme_path.write_text(injected, encoding="utf-8")


if __name__ == "__main__":
    tree = build_tree(REPO_ROOT)
    inject_into_readme(README, tree)
    print("README.md updated with project tree.")
