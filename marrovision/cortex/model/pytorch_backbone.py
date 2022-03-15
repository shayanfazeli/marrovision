from typing import Dict, Any, List
import numpy
import torch
import torch.nn
import torch.nn.functional
import torchvision.models
import marrovision.cortex.optimization.loss as loss_lib
from marrovision.cortex.model.base import ModelBase


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ClassifierWithTorchvisionBackbone(ModelBase):
    def __init__(self, *args, **kwargs):
        super(ClassifierWithTorchvisionBackbone, self).__init__(*args, **kwargs)

    def building_blocks(self) -> None:
        self.softmax = torch.nn.Softmax(dim=-1)
        model_name = self.config['backbone']['type']
        model_args = self.config['backbone']['args']
        model = getattr(torchvision.models, model_name)(**model_args)
        model.fc = Identity()
        self.add_module('backbone', model)
        head_modules = []
        for i in range(len(self.config['head']['modules'])):
            head_modules.append(getattr(torch.nn, self.config['head']['modules'][i]['type'])(**self.config['head']['modules'][i]['args']))
        self.add_module('head', torch.nn.Sequential(*head_modules))
        self.criteria = {loss_module['name']: getattr(loss_lib, loss_module['type'])(**loss_module['args']) for loss_module in self.config['loss']}
        self.criteria_weights = {loss_module['name']: loss_module['weight'] for loss_module in self.config['loss']}

    def initialize_weights(self) -> None:
        pass

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch['image'] = batch['image'].to(self.device)
        batch['label_index'] = batch['label_index'].to(self.device)
        return batch

    def loss(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        loss_outputs = {}
        for k in self.criteria_weights.keys():
            loss_outputs[k] = self.criteria_weights[k] * self.criteria[k](model_outputs['logits'], batch['label_index'])
        return loss_outputs

    def inference_train(self, batch: Any) -> Dict[str, torch.Tensor]:
        model_outputs = dict()
        model_outputs['logits'] = self.backbone(batch['image'])
        model_outputs['y_score'] = self.softmax(model_outputs['logits'])
        model_outputs['y_hat'] = torch.argmax(model_outputs['y_score'], dim=-1)
        model_outputs['label_index'] = batch['label_index']
        return model_outputs

    def inference_eval(self, batch: Any) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.inference_train(batch)
