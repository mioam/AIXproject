import torch
import utils

contents = torch.load('./datasets/pre/content.pt')
print(utils.extract_plaintext(contents[294582]))
print(utils.extract_plaintext(contents[250411]))

