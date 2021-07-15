import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from nibdataloader import DataLoaderImg
from modelunetds import UNetds
from unetdstrain import train
from unetdseval import test


writer = SummaryWriter('runs/unetds14juli')
data_folder = '/scratch/visual/esirin/data'
annotation_folder = '/lumbar_vertebra/labels_all_lumbar_centered/'
image_folder = '/lumbar_vertebra/images_all_lumbar_centered'
batch_size = 1
dataset = DataLoaderImg(data_folder + annotation_folder, data_folder + image_folder)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

EPOCH = 120
learning_rate = 1e-3
momentum = 0.99
in_channels = 1
out_channels = 1
downsapmling_size = 3

features = []
for i in range(downsapmling_size):
    channel = 64 * (2**i)
    features.append(channel)


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetds(in_channels=in_channels, out_channels=out_channels, features=features).cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCH):
        train(epoch, model, optimizer, criterion, DEVICE, train_dataloader, writer)
        test(epoch, model, DEVICE, test_dataloader, writer)

    writer.close()


if __name__ == "__main__":
    main()