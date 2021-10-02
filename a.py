import torch

a = torch.load('./datasets/feature/entity.pt')
total_dict = {}
for x in a:
    for key, val in x.items():
        # print(key, val)
        for x in val:
            if x[0] in total_dict:
                total_dict[x[0]] += x[1]
            else:
                total_dict[x[0]] = x[1]

total_dict = list(total_dict.items())
total_dict.sort(key = lambda x: -x[1])
print(total_dict[:100])


