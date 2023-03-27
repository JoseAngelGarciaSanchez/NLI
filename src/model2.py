import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from transformers.callbacks import EarlyStoppingCallback, ModelCheckpoint


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

    def fit(self, X, y, patience=3):
        
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

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {'accuracy': acc, 'f1_score': f1}

        # Define the callback functions
        early_stopping = EarlyStoppingCallback(early_stopping_patience=patience)
        checkpoint = ModelCheckpoint(
            dirpath=self.output_dir,
            filename='best-checkpoint',
            monitor=self.metric_for_best_model,
            save_top_k=1,
            mode='min'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X,
            eval_dataset=y,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping, checkpoint]
        )

        trainer.train()
        
        # Load the best model from the checkpoint
        best_model_dir = os.path.join(self.output_dir, 'best-checkpoint')
        self.trained_model = self.model.__class__.from_pretrained(best_model_dir)
        
        return self
    
    def predict(self, X):
        # Create a test dataset
        test_dataset = self.transform(X)

        # Use the trained model to make predictions on the test dataset
        predictions = self.trained_model.predict(test_dataset)

        # Convert the model outputs to predicted labels
        predicted_labels = predictions.predictions.argmax(axis=1)

        return predicted_labels
