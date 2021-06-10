import torch
from transformers import get_linear_schedule_with_warmup, AdamW


def step(model, batch, criterion, device):
    targets = batch["label"].to(device)

    outputs, representation = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )

    _, preds = torch.max(outputs, dim=1)

    return (
        {
            "loss": criterion(outputs, targets),
            "accuracy": (preds == targets).float().mean(),
        },
        preds,
        targets,
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