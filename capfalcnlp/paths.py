from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / "resources"
MODELS_DIR = RESOURCES_DIR / "models"
# TODO: Move this to setup or add the folders to the git repo
for dir_path in [MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
FASTTEXT_EMBEDDINGS_DIR = Path(MODELS_DIR) / "fasttext-vectors/"
