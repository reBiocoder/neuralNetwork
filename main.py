import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from model import *
from torchnet import meter
import tqdm
from config import *
from test import *


def train():
    if Config.use_gpu:
        Config.device = t.device("cuda")
    else:
        Config.device = t.device("cpu")
    device = Config.device
    # 获取数据
    datas = np.load("tang.npz")
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=2)

    # 定义模型
    model = PoetryModel(len(word2ix),
                        embedding_dim=Config.embedding_dim,
                        hidden_dim = Config.hidden_dim)
    Configimizer = optim.Adam(model.parameters(),lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    loss_meter = meter.AverageValueMeter()
    # 进行训练
    f = open('result.txt','w')
    for epoch in range(Config.epoch):
        loss_meter.reset()
        for li,data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1,0).contiguous()
            data_ = data_.to(device)
            Configimizer.zero_grad()
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
            loss = criterion(output,target.view(-1))
            loss.backward()
            Configimizer.step()
            loss_meter.add(loss.item())
            # 进行可视化
            if (1+li)%Config.plot_every == 0:
                print("训练损失为%s"%(str(loss_meter.mean)))
                f.write("训练损失为%s\n"%(str(loss_meter.mean)))
                gen_poetry = ''.join(generate(model,"床前明月光",ix2word,word2ix))
                f.write(gen_poetry)
                f.write("\n\n\n")
                f.flush()
        t.save(model.state_dict(),'%s_%s.pth'%(Config.model_prefix,epoch))


if __name__ == '__main__':
    train()