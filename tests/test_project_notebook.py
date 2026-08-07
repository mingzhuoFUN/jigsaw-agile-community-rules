import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "jigsaw_training_method.ipynb"


def test_project_method_notebook_is_portable_and_complete() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20

    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    for required_section in (
        "干净训练数据的构造",
        "生成式模型聊天 SFT",
        "Ettin 编码器分类",
        "规则内排名",
        "Submission 校验",
    ):
        assert required_section in all_source

    assert "google.colab" not in all_source
    assert "userdata.get" not in all_source


def test_project_method_notebook_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
