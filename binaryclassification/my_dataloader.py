
from parser_new import get_parser
from tqdm import tqdm
import numpy as np
import torch
import os
import my_sampler
import my_dataset
from protonet import ProtoNet
import my_loss

query_distance = []


# num_epochs = 1
# num_classes = 10
# batch_size = 100
# learning_rate = 0.001

def init_dataloader(options,mode):
    dataset = my_dataset.MyDataset(mode)
    # sampler = None
    if(mode == 'train'):
        sampler = my_sampler.MySampler(dataset, options.classes_per_it_tr, options.num_support_tr + options.num_query_tr, options.iterations, options.max_num_per_class)
    if(mode == 'dev'):
        sampler = my_sampler.MySampler(dataset, options.classes_per_it_val, options.num_support_val + options.num_query_val, options.iterations, options.max_num_per_class)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

def main():



    options = get_parser().parse_args()
    tr_dataloader = init_dataloader(options,'train')
    dev_dataloader = init_dataloader(options,'dev')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ProtoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    # criterion = loss_test.PrototypicalLoss
    acc_rate_history = 0
    for epoch in range(options.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iterate = iter(tr_dataloader)
        i = 0
        model.train()
        for batch in tqdm(tr_iterate):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss,acc_rate = my_loss.loss(options, model_output, y, 0, 'train')

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                print(acc_rate)
                print(loss)
            i += 1
        dev_iterate = iter(dev_dataloader)
        model.eval()
        for batch in tqdm(dev_iterate):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss,acc_rate = my_loss.loss(options, model_output, y, 0, 'dev')
            print("eval-------------------------------------------")
            print('now {}'.format(acc_rate))
            # print(loss)
            if(acc_rate>acc_rate_history):
                acc_rate_history = acc_rate
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state,'bestmodel.pth')
            print('best {}'.format(acc_rate_history))
            break








if __name__ == '__main__':
    main()