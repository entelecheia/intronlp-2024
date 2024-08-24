from pathlib import Path
from typing import Optional

BOOK_DIR = Path("book")
PROJECTS_DIR = BOOK_DIR / "projects"
toc_file = BOOK_DIR / "_toc.yml"

toc_contents = """
# Table of contents
# Learn more at https://jupyterbook.org/customize/toc.html

format: jb-book
root: index
chapters:
  - file: syllabus/index
"""


def generate_file_entry(
    root: Path,
    file: Path,
    num_indent_space: int = 2,
):
    """
    Generates a file entry for a given file.

    Args:
        root (Path): The root directory of the directory structure.
        file (Path): The file to generate the entry for.
        num_indent_space (int, optional): The number of spaces to use for indentation. Defaults to 2.

    Returns:
        str: The updated contents of the TOC.
    """
    indent = " " * num_indent_space
    rel_path = file.relative_to(root)
    return f"{indent}- file: {rel_path.parent}/{rel_path.stem}\n"


def generate_toc(
    toc_contents: str,
    root: Path,
    section_dir: Optional[Path] = None,
    num_indent_space: int = 2,
):
    """
    Generates a table of contents (TOC) for a given directory structure.

    Args:
        toc_contents (str): The current contents of the TOC.
        root (Path): The root directory of the directory structure.
        section_dir (Optional[Path], optional): The directory to generate the TOC for. Defaults to None.
        num_indent_space (int, optional): The number of spaces to use for indentation. Defaults to 2.

    Returns:
        str: The updated contents of the TOC.
    """
    section_dir = section_dir or root
    _files, _dirs = [], []
    index_path = ""
    for path_ in sorted(section_dir.iterdir()):
        if path_.is_dir():
            _dirs.append(path_)
        elif path_.name == "index.md":
            index_path = path_
        elif path_.suffix in [".md", ".ipynb"]:
            _files.append(path_)
    if index_path:
        toc_contents += generate_file_entry(root, index_path, num_indent_space)

    if len(_files + _dirs) == 0:
        return toc_contents

    num_indent_space += 2
    indent = " " * num_indent_space
    toc_contents += f"{indent}sections:\n"
    for path_ in _files:
        toc_contents += generate_file_entry(root, path_, num_indent_space + 2)

    for path_ in _dirs:
        toc_contents = generate_toc(toc_contents, root, path_, num_indent_space + 2)

    return toc_contents


toc_contents = generate_toc(toc_contents, BOOK_DIR, PROJECTS_DIR)
print(toc_contents)


with open(toc_file, "w") as f:
    f.write(toc_contents)
