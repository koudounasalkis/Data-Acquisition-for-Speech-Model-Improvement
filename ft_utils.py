import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from transformers import TrainingArguments, Trainer

""" Trainer Class """
class WeightedTrainer(Trainer):
    def __init__(self, balancing, **kwargs):
        super().__init__(**kwargs)
        self.balancing = balancing
    
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1)
            )
            
        return (loss, outputs) if return_outputs else loss


""" Define training arguments """ 
def define_training_args(
    output_dir, 
    batch_size, 
    num_steps=500, 
    lr=1.0e-4, 
    gradient_accumulation_steps=1, 
    warmup_steps=0
    ): 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        max_steps=num_steps,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_steps=num_steps,
        save_steps=num_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False)
    return training_args


""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    print('Accuracy: ', acc, 'F1 Macro: ', f1_macro)
    return { 'accuracy': acc, 'f1_macro': f1_macro }