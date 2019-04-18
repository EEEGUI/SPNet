import os
import yaml
import shutil
import torch
import random
from tqdm import tqdm
from icnet import icnet
from utils import get_logger, get_multiply_scale_inputs
from torch.utils import data
from augmentations import *
from cityscapes_loader import cityscapesLoader
from tensorboardX import SummaryWriter


def inference(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device(cfg['training']['device'])

    # dataloader
    testdata = cityscapesLoader("/home/lin/Documents/dataset/Cityscapes/", img_size=(1025, 2049), split='test', is_transform=True)
    testloader = data.DataLoader(testdata, batch_size=cfg['training']['batch_size'])

    # Setup Model
    model = icnet(n_classes=19)
    model.to(device)

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    for i, sample in tqdm(enumerate(testloader)):
        model.eval()
        images = sample['image']
        labels = sample['label']
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        outputs = outputs.data.max(1)[1]
        for j in range(images.shape[0]):
            writer.add_image('image', testdata.img_recover(images[j]), global_step=i * cfg['training']['batch_size'] + j, dataformats='HWC')
            writer.add_image('gt', torch.from_numpy(testdata.decode_segmap(labels.cpu().numpy()[j])), global_step=i * cfg['training']['batch_size'] + j, dataformats='HWC')
            writer.add_image('pred', torch.from_numpy(testdata.decode_segmap(outputs.cpu().numpy()[j])), global_step=i * cfg['training']['batch_size'] + j, dataformats='HWC')

        if i == 10:
            break


if __name__ == "__main__":
    with open("config/icnet-seg.yml") as fp:
        cfg = yaml.load(fp)

    # run_id = random.randint(1, 100000)
    run_id = 309
    logdir = os.path.join("runs", os.path.basename("config/icnet-seg.yml")[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy("config/icnet-seg.yml", logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    inference(cfg, writer, logger)
