import os
import utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10

class TransL_Model(nn.Module):
        def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                # as this is done in nn.CrossEntropyLoss
                for param in self.model.parameters(): # Freeze all parameters
                    param.requires_grad = False

                for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
                    param.requires_grad = True # layer

                for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
                    param.requires_grad = True # layers

        def forward(self, x):
                x = self.model(x)
                return x

def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0



    with torch.no_grad():
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)
            # Forward pass the images through our model
            output_probs = model(X_batch)

            # Compute loss
            loss = loss_criterion(output_probs, Y_batch)

            # Predicted class is the max index over the column dimension
            predictions = output_probs.argmax(dim=1).squeeze()
            Y_batch = Y_batch.squeeze()

            # Update tracking variables
            loss_avg += loss.item()
            total_steps += 1
            total_correct += (predictions == Y_batch).sum().item()
            total_images += predictions.shape[0]
        loss_avg = loss_avg / total_steps
        accuracy = total_correct / total_images

    return loss_avg, accuracy




class Trainer:
    def __init__(self):
        """
                Initialize our trainer class.
                Set hyperparameters, architecture, tracking variables etc.
                """
        # Define hyperparameters
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = TransL_Model()
        # self.model = TransL_Model(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         self.learning_rate)

        # if adam_optimizer:
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
                    Computes the loss/accuracy for all three datasets.
                    Train, validation and test.
                """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
                Checks if validation loss doesn't improve over early_stop_count epochs.
                """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
                Trains the model for [self.epochs] epochs.
                """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                # Compute loss/accuracy for all three datasets.
            # if batch_it % self.validation_check == 0:
            self.validation_epoch()
            # Check early stopping criteria.
            if self.should_early_stop():
                print("Early stopping.")
                return
        torch.save(self.model,"/Users/AndreaViktoria/Documents/Uni/NTNU_SoSe/ComputerVision/ComputerVisionBlatt3.1/assignment3")



if __name__ == "__main__":
    model = torch.load("/Users/AndreaViktoria/Documents/Uni/NTNU_SoSe/ComputerVision/ComputerVisionBlatt3.1/assignment3")
    model.eval()


    trainer = Trainer()
    trainer.train()
    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()

    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])

    validation_loss = np.load("best_model.npy")
    train_loss = np.load("best_model_train.npy")

    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="ResNet18 Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="ResNet18 Training loss")
    plt.plot(validation_loss, label="Selfmade Model Validation loss")
    plt.plot(train_loss, label="Selfmade Model Training loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "comparison.png"))
    plt.show()

