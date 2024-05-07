from numpy import argmin
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.CPL_Net import CPL_Net
from data_loader import train_data



# Training settings
parser = argparse.ArgumentParser(description="PyTorch ResJND")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default="true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--valid_Hz",default=10,type=int,help="Decide how many epochs a validation")

def main():
    global opt, model1
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = train_data('Dataset/train/data_crop',
                           'Dataset/train/label_crop', )

    valid_set = train_data('Dataset/valid/data_crop',
                          'Dataset/valid/label_crop')


    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_set,batch_size=opt.batchSize,shuffle=True)

    print("===> Building model")
    model1 = CPL_Net()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model1 = model1.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model1.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model1.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer1 = optim.SGD(model1.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    print("===> Training")

    #
    loss_valid_list = []
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        loss = train(training_data_loader, optimizer1, model1, criterion, epoch)


        print(f"loss={loss}")
        if epoch %opt.valid_Hz == 0:
            loss_valid = valid(epoch,valid_data_loader,criterion)
            print(f"loss_valid={loss_valid}")
            loss_valid_list.append(loss_valid.item())


    epoch_min = loss_valid_list.index(min(loss_valid_list)) + 1

    save_checkpoint(model1,epoch_min*opt.valid_Hz,loss_valid_list[argmin(loss_valid_list)])
    save_checkpoint(model1,300,loss_valid_list[-1])



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model1, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model1.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model1(input),target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model1.parameters(),opt.clip)
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))

        return loss


def save_checkpoint(model1, epoch,loss_valid):
    model_out_path = "checkpoint/" + "model_epoch_{}_{}.pth".format(epoch, loss_valid)
    state = {"epoch": epoch, "model1": model1}
    if not os.path.exists("Weight"):
        os.makedirs("Weight")

    torch.save(model1.state_dict(),model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def valid(epoch,valid_data_loader,criterion):
    for iteration, batch in enumerate(valid_data_loader, 1):
        input_v, target_v = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input_v = input_v.cuda()
            target_v = target_v.cuda()

        loss_valid = criterion(model1(input_v),target_v) #

    return loss_valid

if __name__ == "__main__":
    main()
