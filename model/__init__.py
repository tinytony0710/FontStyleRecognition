import torch
import torch.nn as nn
import torch.optim as optim

# from .testtest import test
from .module import SwordNet

class Model:
    def __init__(self, prediction_class, device):
        print('__init__')
        self.device = device
        self.model = SwordNet(prediction_class, device=self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train(self, train_dataset, batch_size=16):
        print('train')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        running_loss = 0.0

        n_correct_predictions = 0
        n_samples = 0

        guess_correct = np.zeros(len(train_dataset[0][1]))
        guess_wrong = np.zeros(len(train_dataset[0][1]))
        Y = []
        predictions = []
        for data, labels in train_loader:

            # Forward pass
            outputs = self.model(data)
            Y.append(labels)
            predictions.append(outputs)

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

            n_samples += len(labels)

            for idx, output in enumerate(outputs):
                n_correct_predictions += 1 if torch.argmax(output) == torch.argmax(labels[idx]) else 0
                if torch.argmax(output) == torch.argmax(labels[idx]):
                    guess_correct[torch.argmax(output)] += 1
                else:
                    guess_wrong[torch.argmax(output)] += 1

        # Adjust the learning rate based on the scheduler
        self.scheduler.step()

        # Print statistics
        
        # show your average loss
        accuracy = n_correct_predictions / n_samples
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        print(f"Training loss: {running_loss/len(train_loader):.4f}, accuracy: {accuracy*100:.2f}%")
        for i in range(len(guess_correct)):
            print('type1: ', guess_correct[i], 'vs', guess_wrong[i])


        # After training, you can save the model if desired
        # torch.save(model.state_dict(), "swordnet_model.pth")
        return torch.cat(Y, dim=0).detach().cpu().numpy(), torch.cat(predictions, dim=0).detach().cpu().numpy()
        

    
    def test(self, test_dataset):
        print('test')
        
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        n_correct_predictions = 0
        n_samples = 0

        loss_sum = 0.0
        guess_correct = np.zeros(len(test_dataset[0][1]))
        guess_wrong = np.zeros(len(test_dataset[0][1]))
        Y = []
        predictions = []
        for _x_batch, _y_batch in test_data_loader:
            with torch.no_grad():
                _y_prediction = self.model(_x_batch)
                Y.append(_y_batch)
                predictions.append(_y_prediction)
                loss = self.criterion.forward(_y_prediction, _y_batch)
                loss_sum += loss.item()

                n_correct_predictions += 1 if torch.argmax(_y_prediction) == torch.argmax(_y_batch) else 0
                n_samples += len(_y_batch)

                if torch.argmax(_y_prediction) == torch.argmax(_y_batch):
                    guess_correct[torch.argmax(_y_prediction)] += 1
                else:
                    guess_wrong[torch.argmax(_y_prediction)] += 1

        # show your average loss
        avg_loss = loss_sum / len(test_data_loader)

        accuracy = n_correct_predictions / n_samples
        print(f"Testing loss: {avg_loss:.4f}, accuracy: {accuracy*100:.2f}%")
        for i in range(len(guess_correct)):
            print('type1: ', guess_correct[i], 'vs', guess_wrong[i])

        return torch.cat(Y, dim=0).detach().cpu().numpy(), torch.cat(predictions, dim=0).detach().cpu().numpy()
