import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

class RobertaPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model:str='cross-encoder/nli-roberta-base', train_batch_size:int=8, eval_batch_size:int=32, epochs:int=1, warmup_steps:int=0.1,
                 eval_steps:int=500, metric_for_best_model:str='eval_loss', output_dir:str='./results',
                 load_best_model_at_end:bool=True, random_seed:int=42):
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = output_dir
        self.load_best_model_at_end = load_best_model_at_end
        self.random_seed = random_seed

    def fit(self, X, y):
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='steps',
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.epochs,
            metric_for_best_model=self.metric_for_best_model,
            warmup_steps=self.warmup_steps
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X,
            eval_dataset=y,
            compute_metrics=lambda pred: {"accuracy": accuracy_score(
                pred.label_ids, pred.predictions.argmax(axis=1))}
        )

        self.trainer.train()
        self.trained_model = self.trainer.model
        return self
    
    def predict(self, X, y=None):
        if not hasattr(self, 'trained_model'):
            raise ValueError("The model has not been trained yet. Please call 'fit' first.")

        # Reuse the existing trainer with the trained model
        self.trainer.model = self.trained_model

        predictions = self.trainer.predict(X)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        if y is not None:
            accuracy = accuracy_score(y, predicted_labels)
            return predicted_labels, accuracy
        else:
            return predicted_labels
