"""

SDD1004 - train.py

IMPORTANT : Si vous exécutez le script de nouveau, c'est important de ne pas supprimer out/model_a/
            car c'est le modèle utilisé dans les différents IPYNB de ce projet.

"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import json

import shutil


SEED = 42
MODEL_NAME = "distilbert-base-uncased"


def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)


# label | input_ids | attention_mask (mask c'est quel token est reel et quel token est du padding)
def prepare_train_and_eval_datasets(dataset, tokenizer):
    split_dict = dataset["train"].train_test_split(test_size=0.1, seed=SEED)

    train_ds = split_dict["train"].map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    eval_ds = split_dict["test"].map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    return train_ds, eval_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_training_metrics(trainer):
    """
    Génère un JSON file pour ensuite reproduire un graphique comme celui
    montrer par le prof pendant une des classes (démontre l'évolution du modèle pendant l'entrainement)
    L'utilisation du JSON est dans le IPYNB d'évaluation des performances.
    """
    train_history = trainer.state.log_history

    metrics_dict = {
        "train_logs": [],
        "eval_logs": []
    }

    for log in train_history:
        if "loss" in log and "eval_loss" not in log:
            metrics_dict["train_logs"].append({
                "step": log.get("step"),
                "epoch": log.get("epoch"),
                "loss": log["loss"]
            })

        if "eval_loss" in log:
            metrics_dict["eval_logs"].append({
                "step": log.get("step"),
                "epoch": log.get("epoch"),
                "eval_loss": log["eval_loss"],
                "eval_accuracy": log.get("eval_accuracy")
            })

    metrics_path = Path("out/training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nTraining metrics saved in: {metrics_path}")


def select_epoch_2_checkpoint_and_clean_out_folder():
    """
    - Petite automatisation pour renommer le checkpoint de l'epoch 2 en model_a
    et supprimer tous les autres checkpoints pour s'assurer que tout le projet
    puisse s'exécuter dans l'ordre.

    - L'analyse derrière ce choix est détaillé dans le IPYNB d'évaluation des
    performances du modèle et aussi dans le rapport PDF du projet.
    """
    out_dir = Path("out")
    model_a_dir = out_dir / "model_a"
    target_step = 2814 # Cetait 2814 quand j'avais train avec EPOCH au lieu de STEPS, donc logiquement, le plus proche sera le bom.
    checkpoint_dirs = [
        item for item in out_dir.iterdir()
        if item.is_dir() and item.name.startswith("checkpoint-")
    ]

    if not checkpoint_dirs:
        raise FileNotFoundError("Aucun folder checkpoint-* trouver dans /out")

    epoch_2_checkpoint = min(
        checkpoint_dirs,
        key=lambda p: abs(int(p.name.replace("checkpoint-", "")) - target_step)
    )

    if model_a_dir.exists():
        shutil.rmtree(model_a_dir)

    epoch_2_checkpoint.rename(model_a_dir)

    for item in out_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            shutil.rmtree(item)


def train_transformer_model():
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    train_ds, eval_ds = prepare_train_and_eval_datasets(dataset, tokenizer)

    training_arguments = TrainingArguments(
        output_dir="out",

        num_train_epochs=3, # À rouler avec 3 epoch, mais au final j'ai pris le checkpoint de l'epoch 2 pour le restant du projet (renommé model_a)

        # batch de 4, puis accumulation des gradients pendant 4 steps avant une mise à jour des poids
        per_device_train_batch_size=4, # from 8 to 4 ---> sinon ça plante...  --> nombre d'exemples que le modèle traite en meme temps
        gradient_accumulation_steps=4,

        per_device_eval_batch_size=8,

        fp16=False,

        learning_rate=2e-5, # 0.00002 - vitesse que le modèle ajuste ses poids à chaque update

        # L'OPTIMISEUR:
        #       L'optimiseur n'est pas spécifié, mais après avoir regardé le code de HuggingFace, c'est AdamW (une variante de Adam, améliorer)
        #       (AdamW gère mieux la régularisation par weight decay, pénalise les poids trops grand pour réduire le overfitting)
        weight_decay=0.01,  # Forme de regularisation pour AdamW, ça sert à éviter que les poids du modèle devienne trops grand, ca peut aider à réduire l'overfitting

        # FUNCTION DE LOSS: pas specifier non plus, mais par default la fonction de loss via DistilBERT de HuggingFace c'est CrossEntropyLoss...

        eval_strategy="steps",    # ['no', 'steps', 'epoch', 'best']
        eval_steps=50,
        save_strategy="steps",    # ['no', 'steps', 'epoch', 'best']
        save_steps=50,

        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none", # pas utilisé, mais semble intéressant (À Google: TensorBoard, MLFlow...) - https://huggingface.co/docs/hub/tensorboard
        seed=SEED,
    )

    # Quand on tokenize des textes, ils ont pas toute la meme longueur, donc ajoute des tokens de padding pour avoir la meme longueur dans la batch.... donc ca pad pour que la valeur final soit un multiple de 8
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_training_metrics(trainer)
    select_epoch_2_checkpoint_and_clean_out_folder()


if __name__ == "__main__":
    train_transformer_model()


# Reminder: DistilBERT à 6 blocs transformer, chaque bloc a 12 têtes d'attention, donc un total de 72 têtes d'attention par default