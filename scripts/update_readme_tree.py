from pathlib import Path

START = "<!-- TREE_START -->"
END = "<!-- TREE_END -->"

# Matches `tree -L 2`
MAX_LEVEL = 1  # 1 = only root children, 2 = children + grandchildren

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    "node_modules", ".idea", ".vscode"
}
IGNORE_FILES = {".DS_Store"}

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def should_ignore(p: Path) -> bool:
    if p.is_dir() and p.name in IGNORE_DIRS:
        return True
    if p.is_file() and p.name in IGNORE_FILES:
        return True
    return False


def build_tree(root: Path) -> str:
    lines = [f"{root.name}/"]

    def walk(dir_path: Path, prefix: str, level: int):
        # level = 1 means we're listing children of root
        if level > MAX_LEVEL:
            return

        entries = [p for p in dir_path.iterdir() if not should_ignore(p)]
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))  # dirs first

        for i, p in enumerate(entries):
            last = (i == len(entries) - 1)
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + (p.name + ("/" if p.is_dir() else "")))

            # Only go deeper if next level is allowed
            if p.is_dir() and (level + 1) <= MAX_LEVEL:
                extension = "    " if last else "│   "
                walk(p, prefix + extension, level + 1)

    walk(root, "", 1)
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
    print("README.md updated (max depth = 2 levels).")
