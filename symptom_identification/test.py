import json
import torch
import argparse
import collections
from tqdm import tqdm
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.tokenizer import load_tokenizer
from torch.utils.data import DataLoader
from model.metric import print_classification_report
from model.sngp import mean_field_logits
from model.sngp import SNGP
import torch.nn.functional as F

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('test')
    fold = config['fold']
    mode = config['mode']
    
    # load dataset and tokenizer
    tokenizer = load_tokenizer(config["arch"]["args"]["model_name"])
    data = pd.read_csv(config["data_loader"]["data_dir"])
    with open(config['data_loader']['dict_dir'], 'r') as f:
        s_dict = json.load(f)
    s_dict_inverse = {v: k for k, v in s_dict.items()}    
        
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
    _, X_test, _, y_test, _, _ = module_data.stratified_split(
        data, label_index, disease_index, fold, SEED, n_splits=5, shuffle=True
    )
    
    valid_dataset = module_data.MultiLabelDataLoader(X_test, y_test, tokenizer, max_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False)

    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'sngp':
        backbone = config.init_obj("arch", module_arch)
        model = SNGP(backbone, num_classes=num_labels, device=device)
    else:
        model = config.init_obj("arch", module_arch)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']["calculator"]]

    logger.info('Loading checkpoint: {} ...'.format(config["resume"]))
    checkpoint = torch.load(config["resume"])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        all_labels = np.array([])
        all_preds = np.array([])
        all_symptom_scores = []
        ood_preds = []

        for __builtins__, batch in enumerate(tqdm(valid_dataloader)):
            if mode == 'sngp':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                mean_field_factor = 0.1 # default
                output, cov = model(input_ids, attention_mask, return_gp_cov=True, update_cov=False)
                output = mean_field_logits(output, cov, mean_field_factor=mean_field_factor)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = model(input_ids, attention_mask)

            # adding values
            if mode == 'sngp':
                output = F.sigmoid(output)
            all_labels = np.append(all_labels, labels.tolist())
            all_preds = np.append(all_preds, output.tolist())
            all_labels = torch.Tensor(all_labels.reshape((-1, num_labels))).to(torch.int)
            all_preds = torch.Tensor(all_preds.reshape((-1, num_labels)))
            all_symptom_scores.extend(output.cpu().numpy())

            if mode =='sngp':
                ood_pred = (1. - torch.max(output, dim=-1)[0]).detach().cpu().numpy()
                ood_pred = ood_pred.flatten().tolist()
                ood_preds.extend(ood_pred)

            # computing loss, metrics on test set
            loss = loss_fn(output, labels)
            total_loss += loss.item()

    # total_loss
    n_samples = len(valid_dataloader.sampler)
    log = {'loss': total_loss / n_samples}

    # argument
    threshold = config["metrics"]["threshold"]
    target_name = config["metrics"]["target_name"]

    # metric
    met_log = {}
    for met in metric_fns:
        met_log[met.__name__] = met(threshold, num_labels, all_preds, all_labels)
    log.update(**{"val_" + k: v for k, v in met_log.items()})
    logger.info(log)
    
    # classification_report
    report = (
    print_classification_report(
        threshold, target_name, all_preds, all_labels, output_dict=True
    ))
        
    df = pd.DataFrame(report).transpose()
    df.to_csv(str(config.log_dir)+"/classification_report_fold_"+str(fold)+".csv",index=False)

    ## to csv
    text = X_test # testing
    pred = module_data.revert_label_index(all_preds, threshold, s_dict_inverse)
    label = module_data.revert_label_index(all_labels, threshold, s_dict_inverse)
    logit = all_preds.detach().cpu().numpy().tolist()
    if mode == 'sngp':
        res_df = pd.DataFrame(data={'text': text, 'label': label, 'pred': pred, 'logits': logit, 'ood_preds': ood_preds})
    else:
        res_df = pd.DataFrame(data={'text': text, 'label': label, 'pred': pred})
    res_df.to_csv(str(config.log_dir)+'/error_analysis_fold_'+str(fold)+'.csv', index=False)

        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
                          # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-fold", "--fold"], type=int, target="fold"),
        CustomArgs(["-session", "--session"], type=str, target="session"),
        CustomArgs(["-ckp", "--resume"], type=str, target="resume"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
