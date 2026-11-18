import hashlib
import json
import os
from pathlib import Path

import yaml

DEMO = os.getenv("QA_DEMO", "true").lower() == "true"
TOKEN = os.getenv("QA_TOKEN", "")
ORG = os.getenv("QA_ORG", "local")
MODEL = os.getenv("QA_MODEL", "")
QA_AGENT_MODE = os.getenv("QA_AGENT_MODE", "false").strip().lower() == "true"
QA_AGENT_QUALITY = os.getenv("QA_AGENT_QUALITY", "false").strip().lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_THINK = {"true": True, "false": False, "": None}[os.getenv("OLLAMA_THINK", "").strip().lower()]
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))
MLX_PROMPT_CACHE = os.getenv("MLX_PROMPT_CACHE", "false").strip().lower() == "true"
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "75"))
QA_TASK_TIMEOUT_SECONDS = int(os.getenv("QA_TASK_TIMEOUT_SECONDS", "960"))
PROMPT_VERSION = 7
ENGINE_VERSION = 3
AGENT_VERSION = 13
SEARCH_VERSION = 3
CARDS_DIR = Path(__file__).resolve().parents[1] / "cards"
CARD = yaml.safe_load((CARDS_DIR / "standard.yaml").read_text(encoding="utf-8"))
LEGACY_CARD_FINGERPRINTS = yaml.safe_load((CARDS_DIR / "legacy-card-fingerprints.yaml").read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
