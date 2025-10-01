import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_mnist, ff_model


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition):
    dataset = ff_mnist.FF_MNIST(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(iteration_time, scalar_outputs, partition):
    if scalar_outputs is None:
        return
    
    time_str = f"{iteration_time:.3f}s"
    
    print(f"\tTime: {time_str:<10}", end="")
    
    if partition == "train":
        print(f"\tPeer Normalization Loss: {scalar_outputs['Peer Normalization Loss']:7.4f}", end="")
        print(f"\tBinary losses: ", end="")
        print(", ".join(f"{layer}: {loss:7.4f}" for layer, loss in scalar_outputs["Binary Losses"].items()))
        print(f"\tBinary accuracies: ", end="")
        print(", ".join(f"{layer}: {acc:7.4f}" for layer, acc in scalar_outputs["Binary Accuracies"].items()))
        # for layer, acc in scalar_outputs["Binary Accuracies"].items():
        #     print(f"{layer}: {acc:7.4f}, ", end="")
        print(f"\n\t\t\t\t\t\t\tAccuracy: {scalar_outputs['Accuracy']:7.4f}", end="")
        print(f"\tClassification loss: {scalar_outputs['Classification Loss']:7.4f}")
        
    else:
        if scalar_outputs.get("Mode") == "Classifier head":
            print(f"\tAccuracy: {scalar_outputs['Accuracy']:7.4f}", end="")
            print(f"\tClassification loss: {scalar_outputs['Loss']:7.4f}")
        elif scalar_outputs.get("Mode") == "FF-native":
            print(f"\tAccuracy: {scalar_outputs['Accuracy']:7.4f}", end="")
            print(f"\tAccuracies: ", end="")
            print(", ".join(f"{combo}: {acc:7.4f}" for combo, acc in scalar_outputs["Accuracies"].items()))


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        elif isinstance(value, str):
            continue
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    result_dict[key][subkey] += subvalue / num_steps
                elif isinstance(subvalue, torch.Tensor):
                    result_dict[key][subkey] += subvalue.item() / num_steps
                else:
                    raise ValueError(f"Unsupported type in nested dict: {type(subvalue)}")
        elif isinstance(value, torch.Tensor):
            result_dict[key] += value.item() / num_steps
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return result_dict
