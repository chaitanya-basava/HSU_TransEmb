import os
import torch


def load_checkpoint(
    ckpt,
    model,
    model_dir,
    device,
    _model,
    optimizer=None,
    scheduler=None,
):
    if ckpt:
        checkpoint = torch.load(os.path.join(model_dir, ckpt), map_location=device)

        _model.load_state_dict(checkpoint[model])
        best_weighted_f1 = checkpoint["weighted_f1"]
        start_epoch = checkpoint["epoch"]

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch, best_weighted_f1 = 1, 0.0

    return _model, optimizer, scheduler, best_weighted_f1, start_epoch