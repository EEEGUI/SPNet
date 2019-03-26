import yaml
import torch
import argparse
import timeit
import numpy as np
from utils import depth2img
from icnet import icnet
from dataloader import get_valid_loader
from metrics import runningScore, DepthEstimateScore
torch.backends.cudnn.benchmark = True


def validate(cfg):

    device = torch.device(cfg['training']['device'])

    # Setup Dataloader
    valloader = get_valid_loader(cfg)

    depth_metrics = DepthEstimateScore()

    # Setup Model

    model = icnet()
    state = torch.load(cfg['training']['resume'])
    model.load_state_dict(state['model_state'])
    model.eval()
    model.to(device)

    for i, sample in enumerate(valloader):
        start_time = timeit.default_timer()
        images = sample['image']
        labels = sample['depth']

        images = images.to(device)
        with torch.no_grad():
            pred = model(images)
        depth2img(pred[0][0].cpu().numpy()*80, 'assets/depth.png')
        gt = labels.numpy()

        if cfg['measure_time']:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        depth_metrics.update(gt, pred.cpu().numpy())

    score = depth_metrics.get_scores()

    for k, v in score.items():
        print(k, v)


if __name__ == "__main__":
    with open('config/icnet-depth.yaml') as fp:
        cfg = yaml.load(fp)

    validate(cfg)
