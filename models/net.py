import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.window = 500
        self.step = 400
        self.postnet = nn.Sequential(
            nn.Linear(768, 768)
        )
        self.device = nn.parameter.Parameter(torch.Tensor([0]))

    def forward(self, x):
        x = self.postnet(x)
        return x

    # def forward(self, text):
    #     device = 'cuda'
    #     batch = len(text)
    #     l = []
    #     r = []
    #     outputs = []
    #     for a in text:
    #         l.append(len(outputs))
    #         for i in range(0, len(a), self.step):
    #             s = a[i:i+self.window]
    #             tokens = self.tokenizer(
    #                 s, return_tensors="pt", padding=True).to(device)
    #             outputs.append(self.bert(**tokens).pooler_output[0])
    #             break
    #         r.append(len(outputs))

    #     feature = []
    #     for i in range(batch):
    #         tmp = outputs[l[i]:r[i]]
    #         print(tmp)
    #         tmp = torch.mean(torch.stack(tmp,dim=0), dim=0) # mean
    #         feature.append(tmp)
    #     return feature

    @torch.no_grad()
    def encode(self, text):
        device = self.device.data.device
        outputs = []
        for i in range(0, len(text), self.step):
            s = text[i:i+self.window]
            tokens = self.tokenizer(
                s, return_tensors="pt", padding=True).to(device)
            outputs.append(self.bert(**tokens).pooler_output[0])
        outputs = torch.mean(torch.stack(outputs, dim=0), dim=0)  # mean
        return outputs

    def get_loss(self, a, b):
        device = 'cuda'
        a = torch.stack(a, 0).to(device)
        b = torch.stack(b, 0).to(device)
        # print(a.shape, b.shape)
        a = self.forward(a)
        b = self.forward(b)
        # print(a.shape, b.shape)
        a = F.normalize(a, 2, 1)
        b = F.normalize(b, 2, 1)
        W = torch.matmul(a, b.permute(1, 0))
        mask = torch.eye(4).to(device)
        positive = (mask * W).sum()
        negative = W.sum() - positive
        return positive, negative, negative - positive
