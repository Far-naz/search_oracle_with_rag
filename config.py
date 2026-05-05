import os


ADVISOR_DATA_FILE = "data/cbs_employees.json"
CLEAR_DB = True
LOWEST_SEMANTIC_SCORE = 0.5
ENABLE_LLM_SEARCH = True
ENABLE_ML_INTENT_RECOGNITION = True
INTENT_RULE_CONFIDENCE_THRESHOLD = float(
    os.getenv("INTENT_RULE_CONFIDENCE_THRESHOLD", "0.6")
)
INTENT_ML_CONFIDENCE_THRESHOLD = float(
    os.getenv("INTENT_ML_CONFIDENCE_THRESHOLD", "0.2")
)
INTENT_MODEL_PATH = "models/intent/intent_model.pkl"
INTENT_MODEL_STORE = os.getenv("INTENT_MODEL_STORE", "local")
INTENT_MLFLOW_TRACKING_URI = os.getenv("INTENT_MLFLOW_TRACKING_URI", "")
INTENT_MLFLOW_EXPERIMENT = os.getenv(
    "INTENT_MLFLOW_EXPERIMENT",
    "oracle_intent_recognition",
)
INTENT_MLFLOW_MODEL_URI = os.getenv("INTENT_MLFLOW_MODEL_URI", "")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "by", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "as", "from", "that", "this", "these", "those",
    "it", "its", "into", "about", "over", "under"
}
