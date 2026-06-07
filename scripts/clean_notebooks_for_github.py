import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]


def normalize_source(source):
    if isinstance(source, list):
        return "".join(source), True
    return source or "", False


def denormalize_source(text, was_list):
    if was_list:
        return text.splitlines(keepends=True)
    return text


def sanitize_source(text):
    text = re.sub(r"KGAT_[A-Za-z0-9_\-]+", "KAGGLE_TOKEN_REMOVED", text)
    text = re.sub(r"hf_[A-Za-z0-9]{30,}", "HF_TOKEN_REMOVED", text)
    text = re.sub(
        r"^(\s*)token\s*=\s*[\"']HF_TOKEN_REMOVED[\"']\s*$",
        r'\1token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")',
        text,
        flags=re.MULTILINE,
    )
    return text.replace(
        "# KAGGLE_TOKEN_REMOVED",
        "# Kaggle token removed. Use kagglehub.login() or Kaggle Secrets.",
    )


def clean_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        source_text, was_list = normalize_source(cell.get("source", []))
        source_text = sanitize_source(source_text)
        cell["source"] = denormalize_source(source_text, was_list)

    metadata = notebook.get("metadata", {})
    metadata.pop("widgets", None)
    notebook["metadata"] = metadata

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main():
    for path in sorted(ROOT.glob("*.ipynb")):
        clean_notebook(path)
        print(f"cleaned {path.name}")


if __name__ == "__main__":
    main()
