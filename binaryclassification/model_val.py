from parser_new import get_parser
from tqdm import tqdm
import numpy as np
import torch
import os
import my_sampler
import my_dataset
from protonet import ProtoNet
import my_loss
import  my_dataloader as dl
def main():



    options = get_parser().parse_args()
    dev_dataloader = dl.init_dataloader(options,'dev')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ProtoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    checkpoint = torch.load('bestmodel.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(options.epochs):
        print('=== Epoch: {} ==='.format(epoch))

        dev_iterate = iter(dev_dataloader)
        model.eval()
        for batch in tqdm(dev_iterate):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss,acc_rate = my_loss.loss(options, model_output, y, 0, 'dev')
            print("eval-------------------------------------------")
            print('now {}/130'.format(acc_rate))
            break








if __name__ == '__main__':
    main()