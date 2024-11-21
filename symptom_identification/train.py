import os
import math
import random
import json
import torch
import argparse
import collections
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.tokenizer import load_tokenizer
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model.sngp import SNGP



# fix random seeds for reproducibility
SEED = 42
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main(config):
    logger = config.get_logger("train")
    fold = config['fold']
    mode = config['mode']

    # load dataset and tokenizer
    tokenizer = load_tokenizer(config["arch"]["args"]["model_name"])
    data = pd.read_csv(config["data_loader"]["data_dir"])
    with open(config['data_loader']['dict_dir'], 'r') as f:
        s_dict = json.load(f)    
        
    # arguments
    max_length = config["data_loader"]["max_length"]
    label_cols = config["data_loader"]["label_cols"]
    batch_size = config["data_loader"]["batch_size"]
    shuffle = config["data_loader"]["shuffle"]
    num_labels = config["data_loader"]["num_labels"]
    
    # symptom labels
    label_index = module_data.create_label_index(data, label_cols, num_labels)
    disease_index = module_data.create_label_index(data, 'disease_idx', 5)
    config["metrics"]["target_name"] = list(s_dict.keys())
    print("Target Name : ", list(s_dict.keys()), ">>", len(list(s_dict.keys())))

    # multi-label straitified split
    X_train, X_test, y_train, y_test, _, _ = module_data.stratified_split(
        data, label_index, disease_index, fold, SEED, n_splits=5, shuffle=shuffle
    )
    # Load dataset
    train_dataset = module_data.MultiLabelDataLoader(X_train, y_train, tokenizer, max_length)
    valid_dataset = module_data.MultiLabelDataLoader(X_test, y_test, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

    # build model architecture, then print to console
    device, device_ids = prepare_device(config["n_gpu"])
    if mode == 'sngp':
        backbone = config.init_obj("arch", module_arch)
        model = SNGP(backbone, num_classes=num_labels, device=device)
    else:
        model = config.init_obj("arch", module_arch)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]["calculator"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)

    # set parameters for linear scheduler with warmup
    if config['lr_scheduler']['use']:
        warmup_proportion = config["lr_scheduler"]["warmup_proportion"]
        num_training_steps = len(train_dataloader) * config["trainer"]["epochs"]
        num_warmup_steps = math.ceil(num_training_steps * warmup_proportion)

        # define learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(num_training_steps, num_warmup_steps)
    else: 
        lr_scheduler = None

    # define Trainer instance
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
        lr_scheduler=lr_scheduler,
    )

    # run training
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-fold", "--fold"], type=int, target="fold"),
        CustomArgs(["-session", "--session"], type=str, target="session"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
