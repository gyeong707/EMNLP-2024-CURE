import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import print_classification_report


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        # for metrics
        self.num_class = config["arch"]["args"]["num_labels"]
        self.threshold = config["metrics"]["threshold"]
        self.target_name = config["metrics"]["target_name"]
        self.mode = config['mode']

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        all_labels = np.array([])
        all_preds = np.array([])

        for batch_idx, batch in enumerate(self.data_loader):
            if self.mode == 'sngp':
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                output = self.model(input_ids, attention_mask, update_cov=True) 
            else:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                output = self.model(input_ids, attention_mask)

            self.optimizer.zero_grad()
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # reset precision matrix
            if self.mode == 'sngp':
                self.model.reset_cov()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            all_labels = np.append(labels.tolist(), all_labels)
            all_preds = np.append(output.tolist(), all_preds)
            all_labels = torch.Tensor(all_labels.reshape((-1, self.num_class))).to(
                torch.int
            )
            all_preds = torch.Tensor(all_preds.reshape((-1, self.num_class)))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            if batch_idx == self.len_epoch:
                break

        for met in self.metric_ftns:
            self.train_metrics.update(
                met.__name__, met(self.threshold, self.num_class, all_preds, all_labels)
            )
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            log.update({"report" : self.report})
            
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            all_labels = np.array([])
            all_preds = np.array([])

            for batch_idx, batch in enumerate(self.valid_data_loader):
                if self.mode == 'sngp':
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    output = self.model(input_ids, attention_mask)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    output = self.model(input_ids, attention_mask)
                    
                loss = self.criterion(output, labels)
                all_labels = np.append(labels.tolist(), all_labels)
                all_preds = np.append(output.tolist(), all_preds)
                all_labels = torch.Tensor(all_labels.reshape((-1, self.num_class))).to(
                    torch.int
                )
                all_preds = torch.Tensor(all_preds.reshape((-1, self.num_class)))

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

        for met in self.metric_ftns:
            self.valid_metrics.update(
                met.__name__, met(self.threshold, self.num_class, all_preds, all_labels)
            )
            
        self.report = (
        print_classification_report(
            self.threshold, self.target_name, all_preds, all_labels
        ))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
