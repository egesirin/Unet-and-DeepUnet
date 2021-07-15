import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from diceloss import dice_loss


def train(epoch, model, optimizer, criterion, DEVICE, train_dataloader, writer):
    model.train()
    for batch_idx, (image, target) in enumerate(train_dataloader):
        targets = []
        image = image.to(device=DEVICE)
        target = target.to(device=DEVICE)
        target0 = F.interpolate(target, size=32, mode='trilinear', align_corners=False)
        target0 = (target0 > 0.5).to(torch.float32)
        target1 = F.interpolate(target, size=64, mode='trilinear', align_corners=False)
        target1 = (target1 > 0.5).to(torch.float32)
        targets.append(target0)
        targets.append(target1)
        targets.append(target)
        optimizer.zero_grad()
        outputs = model(image)
        loss = torch.tensor(0.0).to(device=DEVICE)
        mean_closs = torch.tensor(0.0).to(device=DEVICE)
        mean_dloss = torch.tensor(0.0).to(device=DEVICE)
        for idx, output in enumerate(outputs):
            closs = criterion(output, targets[idx])
            mean_closs += closs/len(outputs)
            output = torch.sigmoid(output)
            dloss = 1 - dice_loss(output, targets[idx])
            mean_dloss += dloss/len(outputs)
            loss += (dloss + closs)/len(outputs)
        #mean_closs = mean_closs/len(outputs)
        #mean_dloss = mean_dloss/len(outputs)
        #loss = loss/len(outputs)
        loss.backward()
        optimizer.step()
        if batch_idx == 31:
            print(batch_idx)
        writer.add_scalars('Training Losses/Epoch',
                           {'CrossEntropy': mean_closs.item(), 'Dice': mean_dloss.item(), 'Total': loss.item()},
                           global_step=epoch)
    print('Finished training', epoch)
