import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.discriminant_analysis import StandardScaler
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline,  Trainer, TrainingArguments
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


class Model:
    def __init__(self, model_name="zhangtaolab/dnabert2-promoter", task="text-classification"):
        self.model_name = model_name
        self.task = task
        self.model = None  # For classification pipeline
        self.tokenizer = None
        self.pipe = None
        self.load()  # Automatically load model/tokenizer

    def load(self):
        """Load pretrained model and tokenizer for classification pipeline."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        return self
    
    def extract_embeddings(self, texts, batch_size=16, device=None):
        """
        Returns dense embeddings for a list of sequences.
        Uses the [CLS] token representation from the base AutoModel.
        """
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load base model (not sequence classification) for embeddings
        base_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        base_model.to(device)
        base_model.eval()

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
                batch_texts = texts[i:i+batch_size]
                enc = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = base_model(**enc)

                # Fix for models that return tuple instead of BaseModelOutput
                last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

        return np.vstack(embeddings)

    def build_pipeline(self, top_k=None):
        """Builds Hugging Face inference pipeline."""
        if self.model is None or self.tokenizer is None:
            self.load()
        self.pipe = pipeline(
            self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            top_k=top_k
        )
        return self.pipe

    def predict(self, sequences):
        """Run predictions on list of sequences."""
        if not self.pipe:
            self.build_pipeline()
        results = self.pipe(sequences)
        # Convert predicted label to numeric form (LABEL_0, LABEL_1, etc.)
        preds = [r[0]["label"].replace("LABEL_", "") for r in results]
        return [int(p) for p in preds]

    @staticmethod
    def get_classifiers():
        """Return a dictionary of ML classifiers to test, with scaling where needed."""
        return {
                "SVM": make_pipeline(StandardScaler(), PCA(n_components=64), CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42))),
                # "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
                # "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)),
                # "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
                # Tree-based models do not need scaling
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }
    
    def preprocess(self, dataset, max_length=200):
            """Tokenize dataset sequences"""
            def _tokenize(examples):
                return self.tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=max_length)
            return dataset.map(_tokenize, batched=True)

    def compute_metrics(self, p):
        # Handle tuple output (logits,)
        logits = p.predictions
        if isinstance(logits, tuple):
            logits = logits[0]  # take first element
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    def get_trainer(self, train_dataset, val_dataset, output_dir="./results", lr=0.01, batch_size=16, epochs=5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size*4,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        return trainer