import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW

_sofmax = nn.Softmax(dim=1)


def get_outputs(model, batch, device):
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )


def step(model, batch, criterion, device, test_mode = False):
    

    outputs, representation = get_outputs(model, batch, device)

    soft_outputs = _sofmax(outputs)
    probabilities, preds = torch.max(soft_outputs, dim=1)

    if(test_mode):
        return (
        preds,
        probabilities,
    )
    else:
        targets = batch["label"].to(device)
        return (
        {
            "loss": criterion(outputs, targets),
            "accuracy": (preds == targets).float().mean(),
        },
        preds,
        targets,
        probabilities,
        representation,
    )


def configure_optimizers(model, dataloader, lr, epochs):
    optim = AdamW(
        model.parameters(),
        lr=float(lr),
        correct_bias=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * epochs,
    )
    return optim, scheduler
