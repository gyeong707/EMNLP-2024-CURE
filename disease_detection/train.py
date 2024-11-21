import os
import ast
import pickle
import random
import torch
import math
import numpy as np
import pandas as pd
import argparse
import collections
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.tokenizer import load_tokenizer
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils.util import read_json
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup 
from model.sngp import SNGP


# Fix random seeds for reproducibility
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
    seed_everything(SEED)
    logger = config.get_logger("train")
    mode = config['mode']
    dtype = config['dtype']
    setting = config['setting']
    fold = config["fold"]
    # Session information
    print(mode, dtype, setting, fold)
    print(f"Mode: {config['mode']}")
    print(f"Fold: {config['fold']}")
    print(f"Dtype: {config['dtype']}")
    print(f"Session: {config['session']}")

    # Load dataset
    data_path = config["data_loader"]["data_dir"]
    label_cols = config["data_loader"]["label_cols"]
    num_labels = config["data_loader"]["num_labels"]
    column_name = config["data_loader"]["column_name"]
    data = pd.read_csv(data_path)
    label_index = module_data.create_label_index(data, label_cols, num_labels)

    if mode == 'bert' or mode == 'sngp': # sub-model: bert, bert_with_context
        tokenizer = load_tokenizer(config["arch"]["args"]["model_name"])

    elif mode == 'symp': # sub-model: symp
        pickle_file_path = config["data_loader"]["symptom_dir"]
        with open(pickle_file_path, 'rb') as f:
            states = pickle.load(f)
            data = torch.tensor(np.array(states))
            data = F.sigmoid(data)

    elif mode == 'symp_with_context': # sub-model: symp_with_context
        symptom_file_path = config["data_loader"]["symptom_dir"]
        context_file_path = config["data_loader"]["factor_dir"]
        with open(symptom_file_path, 'rb') as f:
            symptom = pickle.load(f)
            symptom = torch.tensor(np.array(symptom))
            symptom = F.sigmoid(symptom) 
        with open(context_file_path, 'rb') as f:
            context = pickle.load(f)
            context = torch.tensor(np.array(context))

    elif mode == 'ours': # CURE
        tokenizer = load_tokenizer(config["arch"]["args"]["model_name"])
        symptom_file_path = config["data_loader"]["symptom_dir"] # symptom scores
        context_file_path = config["data_loader"]["factor_dir"] # context factors
        uncertainty_file_path = config["data_loader"]["uncertainty_dir"] # uncertainty scores
        with open(symptom_file_path, 'rb') as f:
            symptom = pickle.load(f)
            symptom = torch.tensor(np.array(symptom))
            symptom = F.sigmoid(symptom) 
        with open(context_file_path, 'rb') as f:
            context = pickle.load(f)
            context = torch.tensor(np.array(context))
        with open(uncertainty_file_path, 'rb') as f:
            uncertainty = pickle.load(f)
            uncertainty = torch.tensor(np.array(uncertainty))

        # load gpt_result 
        data['gpt_pred'] = data['gpt_pred'].apply(lambda x: ast.literal_eval(x))
        gpt_pred_list = []
        for pred in data['gpt_pred']:
            gpt_pred_list.append(pred)
        gpt_pred = torch.tensor(np.array(gpt_pred_list)) # result of gpt-4o
        context_text = data['factor_text'].values # context factor in text form

    
    # Multi-label straitified split
    X_train, X_test, y_train, y_test, train_index, test_index = module_data.multilabel_stratified_split(
        data, label_index, SEED, fold, n_splits=5, shuffle=True, mode=mode, column_name=column_name)

    # Define dataset
    if mode == 'bert' or mode == 'sngp':
        max_length = config["data_loader"]["max_length"]
        train_dataset = module_data.BertDataset(X_train, y_train, tokenizer, max_length)
        valid_dataset = module_data.BertDataset(X_test, y_test, tokenizer, max_length)
    
    if mode == 'symp' or mode == 'phq9':
        train_dataset = module_data.SymptomDataset(X_train, y_train)
        valid_dataset = module_data.SymptomDataset(X_test, y_test)

    elif mode == 'symp_with_context':
        X_train_symptom = symptom[train_index]
        X_train_context = context[train_index]
        X_test_symptom = symptom[test_index]
        X_test_context = context[test_index]
        train_dataset = module_data.SymptomFactorDataset(X_train_symptom, X_train_context, y_train)
        valid_dataset = module_data.SymptomFactorDataset(X_test_symptom, X_test_context, y_test)

    elif mode == 'ours': # cure
        max_length = config["data_loader"]["max_length"]
        # X_train [text, symptom, context, context_text, uncertainty, gpt_pred]
        X_train_symptom = symptom[train_index]
        X_train_context = context[train_index]
        X_train_context_text = context_text[train_index]
        X_train_uncertainty = uncertainty[train_index]
        X_train_gpt = gpt_pred[train_index]
        # X_test [text, symptom, context, context_text, uncertainty, gpt_pred]
        X_test_symptom = symptom[test_index]
        X_test_context = context[test_index]
        X_test_context_text = context_text[test_index]
        X_test_uncertainty = uncertainty[test_index]
        X_test_gpt = gpt_pred[test_index]
        # Define dataset
        train_dataset = module_data.OursDataset(tokenizer, max_length, X_train, X_train_symptom, X_train_context, X_train_context_text, X_train_uncertainty, X_train_gpt, y_train)
        valid_dataset = module_data.OursDataset(tokenizer, max_length, X_test, X_test_symptom, X_test_context, X_test_context_text, X_test_uncertainty, X_test_gpt, y_test)
       
    # Define dataloader
    batch_size = config["data_loader"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

    # Build model architecture, then print to console
    device, device_ids = prepare_device(config["n_gpu"])
    if mode == 'symp' or mode == 'symp_with_context':
        device, device_ids = 'cpu', []
        
    if mode == 'ours':
        fold                = config["fold"]
        args_arch           = config['arch']['args']
        target_models       = args_arch["target_models"]
        target_uncertainty  = args_arch["target_uncertainty"]
        num_labels          = args_arch["num_labels"]
        dropout_ratio       = args_arch["dropout_ratio"]
        include_gpt         = args_arch['include_gpt']
        hidden_dim          = args_arch["hidden_dim"]
        num_models          = len(target_models)
        predictors          = []
        for model_type in target_models:
            predictors.append(module_data.load_models(model_type, num_labels, fold, device))
        model = module_arch.CURE(target_models, target_uncertainty, predictors, 
                                 hidden_dim, num_labels, num_models, dropout_ratio, include_gpt)

    elif mode == 'sngp':
        backbone = config.init_obj("arch", module_arch)
        model = SNGP(backbone, num_classes=num_labels, device=device)
    else: # mode == 'bert'
        model = config.init_obj("arch", module_arch)
        
    # Prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    logger.info(model)

    # Get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]["calculator"]]
    
    # Set parameters for linear scheduler with warmup
    if config['lr_scheduler']['use']:
        warmup_proportion = config["lr_scheduler"]["warmup_proportion"]
        num_training_steps = len(train_dataloader) * config["trainer"]["epochs"]
        num_warmup_steps = math.ceil(num_training_steps * warmup_proportion)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else: 
        lr_scheduler = None
        
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

    # Run training!
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["-lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["-bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-fold", "--fold"], type=int, target="fold"),
        CustomArgs(["-session", "--session"], type=str, target="session"),
        CustomArgs(["-data", "--dataset"], type=str, target="data_loader;data_dir"),
        CustomArgs(["-symptom", "--symptom"], type=str, target="data_loader;symptom_dir"),
        CustomArgs(["-uncertainty", "--uncertainty"], type=str, target="data_loader;uncertainty_dir"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
