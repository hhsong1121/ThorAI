import os

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

MODEL_PATH = os.getenv("MLX_MODEL_PATH", "mlx-community/Qwen2.5-7B-Instruct-4bit")
ADAPTER_PATH = os.getenv("MLX_ADAPTER_PATH", "./adapters")
QDRANT_DB_PATH = os.getenv("QDRANT_DB_PATH", "../database/qdrant_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
VINDR_WEIGHTS_PATH = os.getenv(
    "VINDR_WEIGHTS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "vindr_det_epoch_10.pth"),
)
