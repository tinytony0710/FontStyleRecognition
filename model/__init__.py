import torch
import torch.nn as nn
import torch.optim as optim

# from .testtest import test
from .module import SwordNet

class Model:
    def __init__(self, prediction_class, device):
        print('__init__')
        self.model = SwordNet(prediction_class, device=device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train(self, train_dataset, batch_size=16):
        print('train')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        num_epochs = 16
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for data, labels in train_loader:

                # Forward pass
                outputs = self.model(data)

                # Compute the loss
                # print(outputs.shape)
                # print(labels.shape)
                loss = self.criterion.forward(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()  # Zero the gradients before the backward pass
                loss.backward()
                self.optimizer.step()

                # Track the loss
                # print(loss)
                running_loss += loss.item()

            # Adjust the learning rate based on the scheduler
            self.scheduler.step()

            # Print statistics
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


        # After training, you can save the model if desired
        # torch.save(model.state_dict(), "swordnet_model.pth")

        pass
    
    def test(self, test_dataset):
        print('test')
        
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        n_correct_predictions = 0
        n_samples = 0

        loss_sum = 0.0
        for _x_batch, _y_batch in test_data_loader:
            with torch.no_grad():
                # TODO: calculate the loss and accuracy here
                _y_prediction = self.model(_x_batch)
                loss = self.criterion.forward(_y_prediction, _y_batch)
                loss_sum += loss.item()

                n_correct_predictions += 1 if torch.argmax(_y_prediction) == torch.argmax(_y_batch) else 0
                n_samples += len(_y_batch)

        # show your average loss
        avg_loss = loss_sum / len(test_data_loader)

        accuracy = n_correct_predictions / n_samples
        print(f"Testing loss: {avg_loss:.4f}, accuracy: {accuracy*100:.2f}%")
