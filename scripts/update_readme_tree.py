from pathlib import Path

START = "<!-- TREE_START -->"
END = "<!-- TREE_END -->"

MAX_DEPTH = 2

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    "node_modules", ".idea", ".vscode"
}
IGNORE_FILES = {".DS_Store"}

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def should_ignore(path: Path) -> bool:
    return (
        path.name in IGNORE_DIRS
        or path.name in IGNORE_FILES
    )


def build_tree(root: Path) -> str:
    lines = [f"{root.name}/"]

    def walk(dir_path: Path, prefix: str, depth: int):
        if depth >= MAX_DEPTH:
            return

        entries = [p for p in dir_path.iterdir() if not should_ignore(p)]
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

        for i, p in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + (p.name + ("/" if p.is_dir() else "")))

            if p.is_dir():
                walk(p, prefix + ("    " if last else "│   "), depth + 1)

    walk(root, "", 0)
    return "\n".join(lines)


def inject_into_readme():
    content = README.read_text(encoding="utf-8")

    if START not in content or END not in content:
        raise RuntimeError("README.md must contain TREE_START / TREE_END markers")

    before = content.split(START)[0]
    after = content.split(END)[1]

    tree = build_tree(REPO_ROOT)

    README.write_text(
        before
        + START
        + "\n\n```text\n"
        + tree
        + "\n```\n\n"
        + END
        + after,
        encoding="utf-8",
    )


if __name__ == "__main__":
    inject_into_readme()
    print("README.md updated (depth = 2)")
