from time import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Model
from datasets import fontImageGenerator, labeledDataset, ImageDataset
from utils.loader import loadText, loadXsl
from utils.evaluater import evaluate
from utils.stringModifier import excludeCharacter, randomCharacterGenerator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 32
    np.random.seed(seed)
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu

    
    img_height = 96  # image height
    img_width = 96  # image width
    img_shape = (img_height, img_width)

    partial = loadText('assets/部首字.txt')
    thousand = loadText('assets/千字帖.txt')
    thousand = excludeCharacter(thousand, exclusive=partial)
    common = loadXsl('assets/common_character.xls')
    common = excludeCharacter(common, exclusive=partial+thousand)
    common = randomCharacterGenerator(common, size=300)


    fontImages_partial = fontImageGenerator(partial, 'assets/fonts/', 'datasets/fontImages/partial/', img_shape)
    fontImages_thousand = fontImageGenerator(thousand, 'assets/fonts/', 'datasets/fontImages/thousand/', img_shape)
    fontImages_common = fontImageGenerator(common, 'assets/fonts/', 'datasets/fontImages/common/', img_shape)
    fontStyleNum = len(fontImages_partial)

    X_partial, Y_partial = labeledDataset(fontImages_partial)
    X_thousand, Y_thousand = labeledDataset(fontImages_thousand)
    X_common, Y_common = labeledDataset(fontImages_common)

    partial_dataset = ImageDataset(X_partial, Y_partial, device)
    thousand_dataset = ImageDataset(X_thousand, Y_thousand, device)
    common_dataset = ImageDataset(X_common, Y_common, device)

    num_epochs = 20

    m1 = Model(fontStyleNum, device)
    m2 = Model(fontStyleNum, device)
    
    print('Our model')
    our_evaluations_train = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auc': []
    }
    our_evaluations_test = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auc': []
    }

    total_time = 0.0
    our_cost_time = []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] start: ")
        t1 = time()
        Y, predictions_prob = m1.train(partial_dataset, batch_size=2)
        t2 = time()
        total_time += t2 - t1
        our_cost_time.append(total_time)

        predictions = np.zeros_like(predictions_prob)
        for i, prob in enumerate(predictions_prob):
            predictions[i] = np.where(prob == np.max(prob), 1, 0)
        evaluation = evaluate(Y, predictions, predictions_prob)
        our_evaluations_train['accuracy'].append(evaluation['accuracy'])
        our_evaluations_train['f1'].append(evaluation['f1'])
        our_evaluations_train['precision'].append(evaluation['precision'])
        our_evaluations_train['recall'].append(evaluation['recall'])
        our_evaluations_train['auc'].append(evaluation['auc'])

        Y, predictions_prob = m1.test(common_dataset)

        predictions = np.zeros_like(predictions_prob)
        for i, prob in enumerate(predictions_prob):
            predictions[i] = np.where(prob == np.max(prob), 1, 0)
        evaluation = evaluate(Y, predictions, predictions_prob)
        our_evaluations_test['accuracy'].append(evaluation['accuracy'])
        our_evaluations_test['f1'].append(evaluation['f1'])
        our_evaluations_test['precision'].append(evaluation['precision'])
        our_evaluations_test['recall'].append(evaluation['recall'])
        our_evaluations_test['auc'].append(evaluation['auc'])
    print('total time: ',total_time)

    print('Their model')
    their_evaluations_train = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auc': []
    }
    their_evaluations_test = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auc': []
    }
    
    total_time = 0.0
    their_cost_time = []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] start: ")
        t1 = time()
        Y, predictions_prob = m2.train(thousand_dataset, batch_size=2)
        t2 = time()
        total_time += t2 - t1
        their_cost_time.append(total_time)

        predictions = np.zeros_like(predictions_prob)
        for i, prob in enumerate(predictions_prob):
            predictions[i] = np.where(prob == np.max(prob), 1, 0)
        evaluation = evaluate(Y, predictions, predictions_prob)
        their_evaluations_train['accuracy'].append(evaluation['accuracy'])
        their_evaluations_train['f1'].append(evaluation['f1'])
        their_evaluations_train['precision'].append(evaluation['precision'])
        their_evaluations_train['recall'].append(evaluation['recall'])
        their_evaluations_train['auc'].append(evaluation['auc'])

        Y, predictions_prob = m2.test(common_dataset)

        predictions = np.zeros_like(predictions_prob)
        for i, prob in enumerate(predictions_prob):
            predictions[i] = np.where(prob == np.max(prob), 1, 0)
        evaluation = evaluate(Y, predictions, predictions_prob)
        their_evaluations_test['accuracy'].append(evaluation['accuracy'])
        their_evaluations_test['f1'].append(evaluation['f1'])
        their_evaluations_test['precision'].append(evaluation['precision'])
        their_evaluations_test['recall'].append(evaluation['recall'])
        their_evaluations_test['auc'].append(evaluation['auc'])
    print('total time: ',total_time)

    for key in our_evaluations_train:
        plt.plot(our_evaluations_train[key], label='our train')
        plt.plot(our_evaluations_test[key], label='our test')
        plt.plot(their_evaluations_train[key], label='their train')
        plt.plot(their_evaluations_test[key], label='their test')
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.grid()
        plt.legend()
        plt.savefig(f'{key}.png')
        plt.show()
    
    plt.plot(our_cost_time, our_evaluations_train['accuracy'], label='our train')
    plt.plot(our_cost_time, our_evaluations_test['accuracy'], label='our test')
    plt.plot(their_cost_time, their_evaluations_train['accuracy'], label='their train')
    plt.plot(their_cost_time, their_evaluations_test['accuracy'], label='their test')
    plt.xlabel('Cost time')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.savefig(f'accuracy_by_time.png')
    plt.show()


if __name__ == '__main__':
    main()
