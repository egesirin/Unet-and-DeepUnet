import torch
from torch.utils.tensorboard import SummaryWriter
from diceloss import dice_loss


def train(epoch, model, optimizer, criterion, DEVICE, train_dataloader, writer):
    model.train()
    for batch_idx, (image, target) in enumerate(train_dataloader):
        image = image.to(device=DEVICE)
        target = target.to(device=DEVICE)
        optimizer.zero_grad()
        output = model(image)
        closs = criterion(output, target)  #crossentropyloss
        output = torch.sigmoid(output)
        dloss = 1 - dice_loss(output, target)  #diceloss
        loss = dloss + closs
        loss.backward()
        optimizer.step()
        writer.add_scalars('Training Losses/Epoch',
                           {'CrossEntropy': closs.item(), 'Dice': dloss.item(), 'Total': loss.item()},
                           global_step=epoch)
        #print(batch_idx)
    print('Finished training', epoch)
