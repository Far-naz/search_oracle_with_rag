from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.search_engines.bm25 import tokenize
from src.intent.types import IntentPrediction
from src.intent.types import SUPPORTED_INTENTS

SupportedModel = Literal[
    "logistic_regression",
    "svm",
    "xgboost",
    "neural_network",
    "bert_finetuned",
]

FeatureExtractor = Literal["tfidf", "word2vec"]


class MLIntentRecognizer:
    def __init__(
        self,
        model_type: SupportedModel = "logistic_regression",
        feature_extractor: FeatureExtractor = "tfidf",
        remove_stopwords: bool = True,
        use_lemmatization: bool = True,
        word2vec_vector_size: int = 100,
        word2vec_window: int = 5,
        word2vec_min_count: int = 1,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.word2vec_vector_size = word2vec_vector_size
        self.word2vec_window = word2vec_window
        self.word2vec_min_count = word2vec_min_count
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.word2vec_model = None
        self.label_encoder = LabelEncoder()
        self.model = self._build_model(model_type=model_type, random_state=random_state)
        self._is_fitted = False

    def _build_model(self, model_type: SupportedModel, random_state: int):
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)

        if model_type == "svm":
            return SVC(probability=True, kernel="linear", random_state=random_state)

        elif model_type == "xgboost":
            return XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
            )

        elif model_type == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=500,
                random_state=random_state,
            )

        elif model_type == "bert_finetuned":
            raise NotImplementedError(
                "Fine-tuned BERT support is planned for a later phase."
            )

        raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, texts: Sequence[str], intents: Sequence[str]) -> None:
        if len(texts) != len(intents):
            raise ValueError("texts and intents must have the same length")

        if not texts:
            raise ValueError("texts cannot be empty")

        invalid = sorted(
            {intent for intent in intents if intent not in SUPPORTED_INTENTS}
        )
        if invalid:
            raise ValueError(
                "Unsupported intent labels: "
                + ", ".join(invalid)
                + ". Allowed labels are: "
                + ", ".join(sorted(SUPPORTED_INTENTS))
            )

        preprocessed = [self.preprocess_text(text) for text in texts]
        X = self._fit_features(preprocessed)
        y = self.label_encoder.fit_transform(intents)
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, text: str) -> IntentPrediction:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained. Call fit() first.")

        preprocessed = self.preprocess_text(text)
        X = self._transform_features([preprocessed])
        label_id = int(self.model.predict(X)[0])
        intent = str(self.label_encoder.inverse_transform([label_id])[0])

        confidence = self._predict_confidence(X)

        return IntentPrediction(intent=intent, confidence=confidence, source="ml")

    def preprocess_text(self, text: str) -> str:
        tokens = tokenize(
            text or "",
            remove_stopwords=self.remove_stopwords,
            use_lemmatization=self.use_lemmatization,
        )
        if tokens:
            return " ".join(tokens)
        return (text or "").strip().lower()

    def _fit_features(self, preprocessed_texts: Sequence[str]):
        if self.feature_extractor == "tfidf":
            return self.vectorizer.fit_transform(preprocessed_texts)

        if self.feature_extractor == "word2vec":
            self.word2vec_model = self._train_word2vec(preprocessed_texts)
            return self._to_word2vec_matrix(preprocessed_texts)

        raise ValueError(f"Unsupported feature_extractor: {self.feature_extractor}")

    def _transform_features(self, preprocessed_texts: Sequence[str]):
        if self.feature_extractor == "tfidf":
            self._ensure_tfidf_vectorizer_is_fitted()
            try:
                return self.vectorizer.transform(preprocessed_texts)
            except NotFittedError as exc:
                raise RuntimeError(
                    "TF-IDF vectorizer is not fitted. Train and save the intent "
                    "model before calling predict(), or regenerate the saved "
                    "intent model artifact."
                ) from exc

        if self.feature_extractor == "word2vec":
            if self.word2vec_model is None:
                raise RuntimeError("Word2Vec model is not trained. Call fit() first.")
            return self._to_word2vec_matrix(preprocessed_texts)

        raise ValueError(f"Unsupported feature_extractor: {self.feature_extractor}")

    def _ensure_tfidf_vectorizer_is_fitted(self) -> None:
        try:
            check_is_fitted(self.vectorizer, "vocabulary_")
            check_is_fitted(self.vectorizer, "idf_")
        except NotFittedError as exc:
            raise RuntimeError(
                "TF-IDF vectorizer is not fitted. Train and save the intent "
                "model before calling predict(), or regenerate the saved "
                "intent model artifact."
            ) from exc

    def _train_word2vec(self, preprocessed_texts: Sequence[str]):
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "gensim is required for feature_extractor='word2vec'. Install it with pip install gensim."
            ) from exc

        sentences = [
            text.split() if text else ["__empty__"] for text in preprocessed_texts
        ]
        return Word2Vec(
            sentences=sentences,
            vector_size=self.word2vec_vector_size,
            window=self.word2vec_window,
            min_count=self.word2vec_min_count,
            workers=1,
            sg=1,
            seed=self.random_state,
        )

    def _to_word2vec_matrix(self, preprocessed_texts: Sequence[str]) -> np.ndarray:
        if self.word2vec_model is None:
            raise RuntimeError("Word2Vec model is not trained. Call fit() first.")

        matrix = []
        for text in preprocessed_texts:
            tokens = text.split()
            vectors = [
                self.word2vec_model.wv[token]
                for token in tokens
                if token in self.word2vec_model.wv
            ]

            if not vectors:
                matrix.append(np.zeros(self.word2vec_vector_size, dtype=np.float32))
                continue

            matrix.append(np.mean(vectors, axis=0))

        return np.array(matrix)

    def _predict_confidence(self, X) -> float:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            return float(max(probabilities[0]))

        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            if hasattr(decision, "__len__") and len(decision.shape) > 1:
                row = decision[0]
            else:
                row = [float(decision[0])]
            top = max(row)
            shifted = [value - top for value in row]
            exps = [2.718281828**value for value in shifted]
            denom = sum(exps) or 1.0
            return float(max(exps) / denom)

        return 0.5

    def save(self, model_path: str | Path) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained. Call fit() before save().")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": self.model_type,
            "feature_extractor": self.feature_extractor,
            "remove_stopwords": self.remove_stopwords,
            "use_lemmatization": self.use_lemmatization,
            "word2vec_vector_size": self.word2vec_vector_size,
            "word2vec_window": self.word2vec_window,
            "word2vec_min_count": self.word2vec_min_count,
            "random_state": self.random_state,
            "vectorizer": self.vectorizer,
            "word2vec_model": self.word2vec_model,
            "label_encoder": self.label_encoder,
            "model": self.model,
            "is_fitted": self._is_fitted,
        }

        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, model_path: str | Path) -> "MLIntentRecognizer":
        path = Path(model_path)
        with path.open("rb") as file:
            payload = pickle.load(file)

        instance = cls(
            model_type=payload["model_type"],
            feature_extractor=payload.get("feature_extractor", "tfidf"),
            remove_stopwords=payload.get("remove_stopwords", True),
            use_lemmatization=payload.get("use_lemmatization", True),
            word2vec_vector_size=payload.get("word2vec_vector_size", 100),
            word2vec_window=payload.get("word2vec_window", 5),
            word2vec_min_count=payload.get("word2vec_min_count", 1),
            random_state=payload["random_state"],
        )
        instance.vectorizer = payload["vectorizer"]
        instance.word2vec_model = payload.get("word2vec_model")
        instance.label_encoder = payload["label_encoder"]
        instance.model = payload["model"]
        instance._is_fitted = payload["is_fitted"]
        return instance
