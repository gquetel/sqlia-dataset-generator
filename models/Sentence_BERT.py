from pathlib import Path
import evaluate
import torch
import transformers
import pandas as pd
import numpy as np
from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)

from torch.utils.data import Dataset

# We implement this: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
# This notebook is similar and with more details:
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb#scrollTo=71pt6N0eIrJo
# Roberta API: https://huggingface.co/docs/transformers/model_doc/roberta


class CustomSQLIADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RobertaTokenizerFast):
        encodings = tokenizer(
            df["full_query"].tolist(),
            truncation=True,
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Return columns required by RobertaModel.forward()

        https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel.forward
        - intput_ids: indices of input sequence tokens in the vocab (given by RobertaTokenizerFast)
        - attention_mask: ids of padding tokens to avoid performing attention on them.
        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


class CustomBERT:
    def __init__(
        self,
        device: torch.device,
        model_name: str,
        project_paths,
        bert_model: str = "ehsanaghaei/SecureBERT",
        batch_size: int = 16,
        lr: int = 2e-5,
        epochs: int = 5,
        weight_decay: int = 0.01,
    ):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.bert_model = bert_model
        self.weight_decay = weight_decay
        self.model_name = model_name
        self._checkpoints_dir = (
            project_paths.models_path
            + "BERT-models/"
            + self.model_name
            + "/checkpoints"
        )
        self._output_dir = project_paths.models_path + "BERT-models/" + self.model_name

        Path(self._checkpoints_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)

        self.dataset_test = None
        self.dataset_train = None

        self.id2label = {0: 0, 1: 1}
        self.label2id = {0: 0, 1: 1}

        self.model = transformers.RobertaForSequenceClassification.from_pretrained(
            self.bert_model,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

    def _compute_metric(self, eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return self._f1_metric.compute(predictions=preds, references=labels)

    def train(self, save_models: bool = False):
        # Data collator that will dynamically pad the inputs received.
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self._f1_metric = evaluate.load("f1")

        training_args = TrainingArguments(
            output_dir=self._checkpoints_dir,
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            eval_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metric,
        )
        self.trainer = trainer
        trainer.train()

        if save_models:
            trainer.save_model(output_dir=self._output_dir)

    def set_dataloader_test(self, df: pd.DataFrame):
        self.dataset_test = CustomSQLIADataset(df=df, tokenizer=self.tokenizer)

    def set_dataloader_train(self, df: pd.DataFrame):
        self.dataset_train = CustomSQLIADataset(df=df, tokenizer=self.tokenizer)

    def predict(self) -> tuple[list, list]:
        # trainer.predict() doc:
        # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.predict
        nt = self.trainer.predict(test_dataset=self.dataset_test)
        preds = np.argmax(nt.predictions, axis=1)

        # Return target, preds.
        return nt.label_ids, preds

    def predict_probas(self) -> tuple[list, list]:
        nt = self.trainer.predict(test_dataset=self.dataset_test)
        return nt.label_ids, nt.predictions
