import os
import ast
import pickle
import random
import torch
import json
import numpy as np
import pandas as pd
import argparse
import collections
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from tqdm import tqdm
from model.tokenizer import load_tokenizer
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils.util import read_json
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.sngp import mean_field_logits
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
    logger = config.get_logger('test')
    mode = config['mode']
    dtype = config['dtype']
    fold = config["fold"]
    # Check config.args
    print(f"Mode: {config['mode']}")
    print(f"Fold: {config['fold']}")
    print(f"Dtype: {config['dtype']}")
    print(f"Session: {config['session']}")
    print(f"Checkpoint: {config['resume']}")

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
            symptom = pickle.load(f)
            symptom = torch.tensor(np.array(symptom))
            data = symptom
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
        valid_dataset = module_data.BertDataset(X_test, y_test, tokenizer, max_length)
    
    if mode == 'symp' or mode == 'phq9':
        valid_dataset = module_data.SymptomDataset(X_test, y_test)

    elif mode == 'symp_with_context':
        X_test_symptom = symptom[test_index]
        X_test_context = context[test_index]
        valid_dataset = module_data.SymptomFactorDataset(X_test_symptom, X_test_context, y_test)

    elif mode == 'ours': # cure
        max_length = config["data_loader"]["max_length"]
        # X_test [text, symptom, context, context_text, uncertainty, gpt_pred]
        X_test_symptom = symptom[test_index]
        X_test_context = context[test_index]
        X_test_context_text = context_text[test_index]
        X_test_uncertainty = uncertainty[test_index]
        X_test_gpt = gpt_pred[test_index]
        # Define dataset
        valid_dataset = module_data.OursDataset(tokenizer, max_length, X_test, X_test_symptom, X_test_context, X_test_context_text, X_test_uncertainty, X_test_gpt, y_test)
       
    # Define dataloader
    batch_size = config["data_loader"]["batch_size"]
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False) 
        
    # Build model architecture, then print to console
    device, device_ids = prepare_device(config["n_gpu"])
    if mode == 'symp' or mode == 'symptom_with_factor':
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

    # Get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]["calculator"]]

    # Load Checkpoints
    logger.info('Loading checkpoint: {} ...'.format(config["resume"]))
    checkpoint = torch.load(config["resume"])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation
    threshold = config["metrics"]["threshold"]
    target_name = config["metrics"]["target_name"]
    total_loss = 0.0
    with torch.no_grad():
        all_labels, all_preds = np.array([]), np.array([])
        ood_preds = []

        for _, batch in enumerate(tqdm(valid_dataloader)):
            if mode == 'sngp':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                mean_field_factor = 0.1 # default
                output, cov = model(input_ids, attention_mask, return_gp_cov=True, update_cov=False)
                output = mean_field_logits(output, cov, mean_field_factor=mean_field_factor)

            elif mode == 'bert':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output, _ = model(input_ids, attention_mask)

            elif mode == 'symp' or mode == 'phq9':
                inputs = batch['symptom'].to(device)
                labels = batch["labels"].to(device)
                output, _ = model(inputs)

            elif mode == 'symp_with_context':
                symptom = batch["symptom"].to(device)
                factor = batch["factor"].to(device)
                labels = batch["labels"].to(device)
                output, _ = model(symptom, factor)

            elif mode == 'ours':
                text_input_ids = batch["text_input_ids"].to(device)
                text_attention_mask = batch["text_attention_mask"].to(device)
                factor_input_ids = batch["factor_input_ids"].to(device)
                factor_attention_mask = batch["factor_attention_mask"].to(device)
                symptom = batch["symptom"].to(device)
                factor = batch["factor"].to(device)
                labels = batch["labels"].to(device)
                gpt_pred = batch["gpt_pred"].to(device)
                uncertainty = batch["uncertainty"].to(device)
                output = model(text_input_ids, text_attention_mask, 
                                    factor_input_ids, factor_attention_mask,
                                    symptom, factor, uncertainty, gpt_pred)
                    
            output = F.sigmoid(output)
            all_labels = np.append(all_labels, labels.tolist())
            all_preds = np.append(all_preds, output.tolist())
            all_labels = torch.Tensor(all_labels.reshape((-1, num_labels))).to(torch.int)
            all_preds = torch.Tensor(all_preds.reshape((-1, num_labels)))

            if mode =='sngp':
                ood_pred = (1. - torch.max(output, dim=-1)[0]).detach().cpu().numpy()
                ood_pred = ood_pred.flatten().tolist()
                ood_preds.extend(ood_pred)

            loss = loss_fn(output, labels)
            total_loss += loss.item()

    # total_loss
    n_samples = len(valid_dataloader.sampler)
    log = {'loss': total_loss / n_samples}

    # metric 
    met_log = {}
    for met in metric_fns:
        met_log[met.__name__] = met(threshold, num_labels, all_preds, all_labels)
    log.update(**{"val_" + k: v for k, v in met_log.items()})
    met_df = pd.DataFrame(met_log, index=[0])
    met_df = met_df.applymap(lambda x: x.item())
    met_df.to_csv(str(config.log_dir)+"/metric_fold_"+str(fold)+".csv")

    # classification_report
    report = (
    module_metric.print_classification_report(
        threshold, target_name, all_preds, all_labels, output_dict=True 
    ))
    logger.info(log)
    
    report_str = json.dumps(report, indent=2)
    report_lines = report_str.split('\n')
    for line in report_lines:
        logger.info(line)
        
    df = pd.DataFrame(report).transpose()
    df.to_csv(str(config.log_dir)+"/classification_report_fold_"+str(fold)+".csv")

    ## to csv
    print(config.log_dir)
    pred = module_data.revert_label_index(all_preds, threshold)
    label = module_data.revert_label_index(all_labels, threshold)
    logit = all_preds.detach().cpu().numpy().tolist()
    
    if mode == 'symp':
        inputs = symptom[test_index]
        inputs = inputs.detach().cpu().numpy().tolist()
    elif mode == 'symp_with_context':
        inputs = factor[test_index]
        inputs = inputs.detach().cpu().numpy().tolist()
    column_name = config["data_loader"]["column_name"]
    data = pd.read_csv(config["data_loader"]["data_dir"])
    text = data[column_name].loc[test_index]

    if mode == 'sngp':
        res_df = pd.DataFrame(data={'text': text, 'label': label, 'pred': pred, 'logits': logit, 'ood_preds': ood_preds})
    elif mode == 'symp' or mode == 'symp_with_context':
        res_df = pd.DataFrame(data={'text': text, 'input': inputs, 'label': label, 'pred': pred, 'logits': logit})
    else:
        res_df = pd.DataFrame(data={'text': text, 'label': label, 'pred': pred, 'logits': logit})

    res_df.to_csv(str(config.log_dir)+'/error_analysis_fold_'+str(fold)+".csv", index=True)
                  

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["-lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["-bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-fold", "--fold"], type=int, target="fold"),
        CustomArgs(["-session", "--session"], type=str, target="session"),
        CustomArgs(["-ckp", "--resume"], type=str, target="resume"),
        CustomArgs(["-data", "--dataset"], type=str, target="data_loader;data_dir"),
        CustomArgs(["-symptom", "--symptom"], type=str, target="data_loader;symptom_dir"),
        CustomArgs(["-uncertainty", "--uncertainty"], type=str, target="data_loader;uncertainty_dir"),
    ]
    config = ConfigParser.from_args(args, options)
    print(config['resume'])
    main(config)



