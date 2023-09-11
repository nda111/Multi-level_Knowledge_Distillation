import os
import timm
import data

import models


def setup_imagenet_head(model_name: str):
    filename = models.get_checkpoint_filename(model_name, 'imagenet')
    if not os.path.exists(filename):
        import torch
        net = __get_model(model_name, pretrained=True)
        head = net.get_classifier()
        torch.save({'head_state_dict': head.state_dict()}, filename)


def get_input_size(model_name: str):
    model_name = model_name.lower()
    return timm.get_pretrained_cfg_value(model_name, 'input_size')[1:]


def __get_model(model_name: str, dataset_name: str, pretrained: bool=True, load_head: bool=False):
    model_name = model_name.strip().lower()
    dataset_name = dataset_name.strip().lower()
    create_params = dict(model_name=model_name)
    
    if dataset_name is not None:
        create_params['num_classes'] = len(data.get_classes(dataset_name))
    create_params['pretrained'] = pretrained
    
    net = timm.create_model(**create_params)
    if load_head:
        import torch
        filename = models.get_checkpoint_filename(model_name, dataset_name)
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Checkpoint for '{model_name}' and '{dataset_name}' was not found. "
                f"Run 'python models/ipts/train_classifier.py --model {model_name} --dataset {dataset_name}' first.")
        
        ckpt = torch.load(filename)
        state_dict = ckpt['head_state_dict']
        net.get_classifier().load_state_dict(state_dict)
    
    return net


def get_teacher(model_name: str, dataset_name: str):
    return __get_model(model_name, dataset_name, pretrained=True, load_head=True)


def get_student(model_name: str, dataset_name: str, pretrained: bool=False):
    return __get_model(model_name, dataset_name, pretrained=pretrained, load_head=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    raise NotImplementedError
