import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from icnet import icnet
from metrics import DepthEstimateScore, averageMeter
from torch import optim
from loss import depth_loss
from utils import get_logger
from schedulers import ConstantLR
from torch.utils import data
from torchvision.transforms import Compose
from dataloader import get_train_loader, get_valid_loader, get_test_loader
from augmentations import *
from cityscapes_loader import cityscapesLoader
from tensorboardX import SummaryWriter


def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device(cfg['training']['device'])

    # Setup Metrics
    scores = DepthEstimateScore()

    # dataloader
    augmentations = Compose([RandomHorizonFlip(0.5), RandomRotate(10)])
    traindata = cityscapesLoader("/home/lin/Documents/dataset/Cityscapes/", img_size=(1025, 2049), split='train', is_transform=True, augmentations=augmentations)
    valdata = cityscapesLoader("/home/lin/Documents/dataset/Cityscapes/", img_size=(1025, 2049), split='val', is_transform=True)
    trainloader = data.DataLoader(traindata, batch_size=cfg['training']['batch_size'])
    valloader = data.DataLoader(valdata, batch_size=cfg['training']['batch_size'])

    # Setup Model
    model = icnet()
    model.to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = ConstantLR(optimizer)

    loss_fn = depth_loss

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_score = -100.0
    i = start_iter
    flag = True

    while i <= cfg["training"]["train_iters"] and flag:
        for sample in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = sample['image']
            labels = sample['depth']
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(labels, outputs[0])

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                model.eval()
                with torch.no_grad():
                    for i_val, sample in tqdm(enumerate(valloader)):
                        images_val = sample['image'].to(device)
                        labels_val = sample['depth'].to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(labels_val, outputs[0])

                        scores.update(labels_val.cpu().numpy(), outputs[0].cpu().numpy())
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score = scores.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                val_loss_meter.reset()
                scores.reset()

                if score["rmse"] >= best_score:
                    best_iou = score["rmse"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    with open("config/icnet-depth.yaml") as fp:
        cfg = yaml.load(fp)

    # run_id = random.randint(1, 100000)
    run_id = 1103
    logdir = os.path.join("runs", os.path.basename("config/icnet-depth.yaml")[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy("config/icnet-depth.yaml", logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
