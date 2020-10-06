from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union, TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from thc.utils.constants import DISTILBERT_TOKENS_MAX_LENGTH

if TYPE_CHECKING:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from transformers import DistilBertTokenizer, DistilBertTokenizerFast

    from thc.models import DistilBertClassifier


@dataclass(frozen=True)
class EpochResults:
    outputs: np.ndarray
    targets: np.ndarray
    average_loss: float


@dataclass(frozen=True)
class TrainTestDataloaders:
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


def train(
    model: DistilBertClassifier,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: Union[DistilBertTokenizer, DistilBertTokenizerFast],
    device: torch.device,
    optimizer: optim.Optimizer,
    objective: nn.modules.loss._Loss,
) -> EpochResults:
    if objective.reduction != "mean":
        return ValueError(
            "`objective` parameter accepts only losses with `reduction='mean'`"
        )

    model = model.to(device)
    model.train()

    outputs = []
    targets = []
    average_loss = 0.0
    for input_batch, target_batch in tqdm(dataloader):
        encoding_batch = tokenizer.batch_encode_plus(
            list(input_batch),
            max_length=DISTILBERT_TOKENS_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids_batch = encoding_batch["input_ids"].to(device)
        attention_mask_batch = encoding_batch["attention_mask"].to(device)

        optimizer.zero_grad()

        output_batch = model(
            input_ids=input_ids_batch, attention_mask=attention_mask_batch
        )
        outputs.append(output_batch.cpu().detach().numpy())

        target_batch = target_batch.view(-1)
        targets.append(target_batch.cpu().detach().numpy())
        target_batch = target_batch.to(device)

        loss = objective(output_batch, target_batch.long())
        average_loss += loss.item() * len(input_batch)
        loss.backward()

        optimizer.step()

    average_loss /= len(dataloader)

    return EpochResults(
        outputs=np.concatenate(outputs, axis=0),
        targets=np.concatenate(targets, axis=0),
        average_loss=average_loss,
    )


def test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: Union[DistilBertTokenizer, DistilBertTokenizerFast],
    device: torch.device,
    objective: nn.modules.loss._Loss,
) -> EpochResults:
    if objective.reduction != "mean":
        return ValueError(
            "`objective` parameter accepts only losses with `reduction='mean'`"
        )

    model = model.to(device)
    model.eval()

    outputs = []
    targets = []
    average_loss = 0.0
    with torch.no_grad():
        for input_batch, target_batch in tqdm(dataloader):
            encoding_batch = tokenizer.batch_encode_plus(
                list(input_batch),
                max_length=DISTILBERT_TOKENS_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids_batch = encoding_batch["input_ids"].to(device)
            attention_mask_batch = encoding_batch["attention_mask"].to(device)

            output_batch = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch
            )
            outputs.append(output_batch.cpu().detach().numpy())

            target_batch = target_batch.view(-1)
            targets.append(target_batch.cpu().detach().numpy())
            target_batch = target_batch.to(device)

            average_loss += objective(
                output_batch, target_batch.long()
            ).item() * len(input_batch)

    average_loss /= len(dataloader)

    return EpochResults(
        outputs=np.concatenate(outputs, axis=0),
        targets=np.concatenate(targets, axis=0),
        average_loss=average_loss,
    )


def run_experiment(
    model: nn.Module,
    dataloaders: TrainTestDataloaders,
    tokenizer: Union[DistilBertTokenizer, DistilBertTokenizerFast],
    device: torch.device,
    optimizer: optim.Optimizer,
    objective: nn.modules.loss._Loss,
    epochs: int = 10,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    for epoch in tqdm(range(epochs)):
        train_epoch_results = train(
            model=model,
            dataloader=dataloaders.train,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            objective=objective,
        )
        test_epoch_results = test(
            model=model,
            dataloader=dataloaders.test,
            tokenizer=tokenizer,
            device=device,
            objective=objective,
        )
        if scheduler is not None:
            scheduler.step()

        if artifacts_dir is not None:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            now = datetime.now()
            torch.save(
                checkpoint,
                Path(artifacts_dir).joinpath(
                    "checkpoint.{}.pth".format(
                        now.strftime("%d-%m-%Y.%H_%M_%S")
                    )
                ),
            )

        if writer is not None:
            writer.add_scalar(
                "avg-loss-train", train_epoch_results.average_loss, epoch
            )
            writer.add_scalar(
                "avg-loss-test", test_epoch_results.average_loss, epoch
            )

            predictions_train = np.argmax(train_epoch_results.outputs, axis=1)
            predictions_test = np.argmax(test_epoch_results.outputs, axis=1)

            writer.add_scalar(
                "micro-f1-train",
                f1_score(
                    y_true=train_epoch_results.targets,
                    y_pred=predictions_train,
                    average="micro",
                ),
                epoch,
            )
            writer.add_scalar(
                "micro-f1-test",
                f1_score(
                    y_true=test_epoch_results.targets,
                    y_pred=predictions_test,
                    average="micro",
                ),
                epoch,
            )

            writer.add_scalar(
                "macro-f1-train",
                f1_score(
                    y_true=train_epoch_results.targets,
                    y_pred=predictions_train,
                    average="macro",
                ),
                epoch,
            )
            writer.add_scalar(
                "macro-f1-test",
                f1_score(
                    y_true=test_epoch_results.targets,
                    y_pred=predictions_test,
                    average="macro",
                ),
                epoch,
            )

            if scheduler is not None:
                writer.add_scalar(
                    "learning-rate", scheduler.get_last_lr()[0], epoch
                )
    if writer is not None:
        writer.close()
