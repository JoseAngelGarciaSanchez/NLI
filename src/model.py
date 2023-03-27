from sklearn.base import BaseEstimator, TransformerMixin
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import random

class TransformerTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, model, train_batch_size=8, eval_batch_size=32, epochs=1, warmup_steps=0.1,
                 eval_steps=500, metric_for_best_model='eval_loss', output_dir='./results',
                 load_best_model_at_end=True, random_seed=42):
        self.model = model
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
        # Train the model on a smaller sample of the training data
        random.seed(self.random_seed)
        train_data_sample = random.sample(X, k=10000)

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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data_sample,
            eval_dataset=y,
            compute_metrics=lambda pred: {"accuracy": accuracy_score(
                pred.label_ids, pred.predictions.argmax(axis=1))}
        )

        trainer.train()
        self.trained_model = trainer.model
        return self

    def transform(self, X):
        # return transformed data
        return X
