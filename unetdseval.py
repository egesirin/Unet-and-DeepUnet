import torch
from diceloss import dice_loss


def test(epoch, model, DEVICE, test_dataloader, writer):
    model.eval()
    dice_score = 0
    with torch.no_grad():
        for image, target in test_dataloader:
            image = image.to(device=DEVICE)
            target = target.to(device=DEVICE)
            output = model(image)[-1]
            prediction = torch.sigmoid(output)
            prediction = (prediction > 0.5).to(torch.float32)
            dice_score += dice_loss(prediction, target)
    writer.add_scalar('DiceScore/Epoch', dice_score.item()/len(test_dataloader), global_step=epoch)
