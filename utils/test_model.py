import os

import torch

from utils.utils import get_logger, is_logging_process
import wandb


def test_model(cfg, model, test_loader, writer):
    logger = get_logger(cfg, os.path.basename(__file__))
    model.net.eval()
    total_test_loss = 0
    test_loop_len = 0
    with torch.no_grad():
        example_images = []
        for model_input, model_target in test_loader:
            output = model.inference(model_input)
            loss_v = model.loss_f(output, model_target.to(cfg.device))
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                torch.distributed.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
            total_test_loss += loss_v.to("cpu").item()
            test_loop_len += 1
            if test_loop_len <10:
                example_images.append(wandb.Image(
                    model_input[0], caption="Pred: {} Truth: {}".format(output.max(1, keepdim=True)[1][0].item(), model_target[0])))

        total_test_loss /= test_loop_len


        if writer is not None:
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
            writer.logging_with_step(example_images, model.step, "Examples")
        if is_logging_process():
            logger.info("Test Loss %.04f at step %d" % (total_test_loss, model.step))
