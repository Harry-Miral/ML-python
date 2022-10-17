from cnn import CnnNet
from torch.utils.data import DataLoader, Subset
from dataset import MyDataset
import torch
import torch.nn.functional as F
import math
import os
from tqdm import tqdm


def Train(k=0):
    if not os.path.exists('./model_dict'):
        os.mkdir('./model_dict')
    print('Start training {}th feature extraction model based on CNN...'.format(k))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MyDataset(k)
    learning_rate = 0.00001

    batch_size = 32
    epoch = 150
    echo_every = 15
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True)

    model = CnnNet(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

    loss_fn = torch.nn.CrossEntropyLoss()
    evl_best = -math.inf
    for epoch in tqdm(range(1, epoch + 1)):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        num_train_correct = 0
        num_train_examples = 0
        for index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            img = batch[0].to(device).to(torch.float32)
            target = batch[1].to(device).to(torch.float).view((img.shape[0]))
            output, feature = model(img)
            loss = loss_fn(output, target.long())
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * img.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                               target)
            num_train_correct += torch.sum(correct).item()
            num_train_examples += correct.shape[0]
            # if index % echo_every == 0 and index > 0:
            #     print(
            #         'Train Epoch: {} [{}/{}]\t  Loss:{:.6f}  Accuracy = {:.2f} '.format(
            #             epoch, index * batch_size, train_size, loss.item(), num_train_correct / num_train_examples))
        training_loss /= train_size
        model.eval()
        with torch.no_grad():
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                img = batch[0].to(device).to(torch.float32)
                target = batch[1].to(device).to(torch.float).view((img.shape[0]))
                output, feature = model(img)
                loss = loss_fn(output, target.long())
                valid_loss += loss.data.item() * img.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                                   target)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= val_size
            correct_rate = num_correct / num_examples
        # print('Epoch: {}, Training Loss: {:.2f}  Validation Loss: {:.2f}  Val_Accuracy = {:.2f}'.format(epoch,
        #                                                                                                 training_loss,
        #                                                                                                 valid_loss,
        #                                                                                                 correct_rate))
        if correct_rate > evl_best:
            evl_best = correct_rate
            torch.save(model, './model_dict/{}th_Net.pkl'.format(k))
            # print('Save best model done!')
