import torch

from argus import Model
from argus.utils import deep_detach, deep_to

from src.models.aux_skip_attention import AuxSkipAttention
from src.losses import OnlyNoisyLqLoss, OnlyNoisyLSoftLoss, BCEMaxOutlierLoss
from src import config


class FreesoundModel(Model):
    nn_module = {
        'AuxSkipAttention': AuxSkipAttention,
    }
    loss = {
        'OnlyNoisyLqLoss': OnlyNoisyLqLoss,
        'OnlyNoisyLSoftLoss': OnlyNoisyLSoftLoss,
        'BCEMaxOutlierLoss': BCEMaxOutlierLoss
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)

        if 'aux' in params:
            self.aux_weights = params['aux']['weights']
        else:
            self.aux_weights = None

        self.use_amp = not config.kernel and 'amp' in params
        if self.use_amp:
            from apex import amp
            self.amp = amp
            self.nn_module, self.optimizer = self.amp.initialize(
                self.nn_module, self.optimizer,
                opt_level=params['amp']['opt_level'],
                keep_batchnorm_fp32=params['amp']['keep_batchnorm_fp32'],
                loss_scale=params['amp']['loss_scale']
            )

    def prepare_batch(self, batch, device):
        input, target, noisy = batch
        input = deep_to(input, device, non_blocking=True)
        target = deep_to(target, device, non_blocking=True)
        noisy = deep_to(noisy, device, non_blocking=True)
        return input, target, noisy

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        input, target, noisy = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        if self.aux_weights is not None:
            loss = 0
            for pred, weight in zip(prediction, self.aux_weights):
                loss += self.loss(pred, target, noisy) * weight
        else:
            loss = self.loss(prediction, target, noisy)
        if self.use_amp:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction[0]),
            'target': target,
            'loss': loss.item(),
            'noisy': noisy
        }

    def val_step(self, batch) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target, noisy = self.prepare_batch(batch, self.device)
            prediction = self.nn_module(input)
            if self.aux_weights is not None:
                loss = 0
                for pred, weight in zip(prediction, self.aux_weights):
                    loss += self.loss(pred, target, noisy) * weight
            else:
                loss = self.loss(prediction, target, noisy)
            return {
                'prediction': self.prediction_transform(prediction[0]),
                'target': target,
                'loss': loss.item(),
                'noisy': noisy
            }

    def predict(self, input):
        assert self.predict_ready()
        with torch.no_grad():
            if self.nn_module.training:
                self.nn_module.eval()
            input = deep_to(input, self.device)
            prediction = self.nn_module(input)
            if self.aux_weights is not None:
                prediction = prediction[0]
            prediction = self.prediction_transform(prediction)
            return prediction
