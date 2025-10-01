import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from src import utils


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        # Create a dict of floats or dicts of floats using defaultdict
        train_results = defaultdict(lambda: defaultdict(float))
        train_results["Mode"] = "Train"
        train_results["Loss"] = 0.0
        train_results["Peer Normalization Loss"] = 0.0
        train_results["Classification Loss"] = 0.0
        # train_results["Binary Losses"] = defaultdict(float)
        # train_results["Binary Accuracies"] = defaultdict(float)
        train_results["Accuracy"] = 0.0
        # train_results["Accuracies"] = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()
            
            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        print(f"Epoch {epoch}:")
        print(f"\ttraining\t", end="")
        print(train_results)
        utils.print_results(time.time() - start_time, train_results, "train")
        start_time = time.time()

        # Validate using both methods.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)
            validate_or_test_ff_native(opt, model, "val", epoch=epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None):
    """Standard downstream classifier validation."""
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(f"\t{partition} (classifier)", end="")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(time.time() - test_time, test_results, partition)
    model.train()


def validate_or_test_ff_native(opt, model, partition, epoch=None):
    """FF-native validation using goodness maximization."""
    test_time = time.time()
    test_results = defaultdict(float)
    test_results["Mode"] = "FF-native"
    test_results["Accuracies"] = defaultdict(float)
    

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(f"\t{partition} (FF-native)\t", end="")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_ff_native_validation(inputs, labels)
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(time.time() - test_time, test_results, partition)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    
    # Final validation using both methods
    validate_or_test(opt, model, "val")
    validate_or_test_ff_native(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")
        validate_or_test_ff_native(opt, model, "test")


if __name__ == "__main__":
    my_main()
