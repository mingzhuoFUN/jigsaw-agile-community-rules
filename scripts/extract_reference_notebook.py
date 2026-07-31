"""Extract every %%writefile payload from the supplied reference notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference/notebook_cells"))
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        first, separator, payload = source.partition("\n")
        if cell.get("cell_type") != "code" or not first.startswith("%%writefile ") or not separator:
            continue
        original_name = first.removeprefix("%%writefile ").strip()
        target = args.output / f"cell_{index:02d}_{Path(original_name).name}"
        target.write_text(payload, encoding="utf-8")
        manifest.append({"cell": index, "original_name": original_name, "extracted_path": target.as_posix()})

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(manifest)} scripts to {args.output}")


if __name__ == "__main__":
    main()
