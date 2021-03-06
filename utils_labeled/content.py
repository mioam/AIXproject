import torch
from utils.extraction import extract_plaintext, getAnHao

contents = torch.load('/mnt/data/mzc/datasets/pre/content.pt')
# print(utils.extract_plaintext(contents[294582]))
# print(utils.extract_plaintext(contents[250411]))
for i in range(20):
    print(getAnHao(extract_plaintext(contents[i])))
# print(contents[214892])
# print(contents[378251])
# print(contents[214887])
# print(contents[169439])


# [('2017', '黔01', '行初', '239')]
# [('2017', '沪0112', '民初', '7352')]
# [('2017', '赣01', '刑更', '2901')]
# [('2017', '陕01', '民终', '5285'), ('2016', '陕0102', '民初', '4141'), ('2016', '陕0102', '民初', '4141')]
# [('2017', '赣01', '民终', '459')]
# [('2017', '黔03', '民终', '1886'), ('2016', '黔0321', '民初', '5885'), ('2016', '黔0321', '民初', '5885')]
# [('2017', '浙0802', '执', '1102'), ('2016', '浙0802', '民初', '4084')]
# [('2016', '辽1223', '执', '142')]
# [('2017', '黔06', '民辖终', '20')]
# [('2017', '黔06', '民终', '376'), ('2016', '黔0626', '民初', '2257')]
# [('2016', '浙0802', '民初', '3390')]
# [('2017', '浙0802', '执', '1127'), ('2016', '浙0802', '民初', '5889')]
# [('2016', '浙0802', '行初', '221'), ('2016', '浙0802', '行初', '164')]
# [('2017', '黔06', '刑初', '13')]
# [('2017', '黔26', '民终', '566'), ('2016', '黔2622', '民初', '820')]
# [('2017', '黔26', '民终', '449')]
# [('2017', '黔26', '民终', '723'), ('2017', '黔2630', '民初', '28'), ('2016', '黔2630', '民初', '405'), ('2016', '黔2630', '民初', '405')]
# [('2016', '黔2626', '民初', '481')]
# [('2017', '黔26', '行初', '9')]
# [('2017', '黔26', '行初', '14')]