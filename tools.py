import torch
from torch.utils.data import DataLoader
import importlib
import numpy as np
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y



root_path = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

data = PartNormalDataset(root=root_path, npoints=2048, split='test', normal_channel=True)
test_dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

num_classes = 2
num_part = 50
num_votes = 3
seg_classes = {'Person': [1], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

''' model loading'''
from log.part_seg.pointnet2_part_seg_msg import pointnet2_part_seg_msg as model

classifier = model.get_model(num_part, normal_channel=True).cuda()
checkpoint = torch.load('./log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    seg_label_to_cat = {}

    for cat in seg_classes.keys():  # {0:Airplane, 1:Airplane, ...49:Table}
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    classifier = classifier.eval()
    for batch_id, (points, label, target) in enumerate(test_dataloader):
        batchsize, num_point, _ = points.size()
        cur_batch_size, NUM_POINT, _ = points.size()
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0],target.size()[1],num_part).cuda()

    for _ in range(num_votes):
        seg_pred, _ = classifier(points, to_categorical(label, num_classes))
        vote_pool += seg_pred

    seg_pred = vote_pool / num_votes  # 求平均？
    cur_pred_val = seg_pred.cpu().data.numpy()
    cur_pred_val_logits = cur_pred_val  # 转存结果->(1,2048,50)
    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # 由(1,2048,50)->(1,2048)
    target = target.cpu().data.numpy()  # target->(1,2048) 1为batch_size大小

    for i in range(cur_batch_size):
        cat = seg_label_to_cat[target[i, 0]]
        logits = cur_pred_val_logits[i, :, :]
        # logits 形状(2048,50) ,50代表num_part, logits[:,seg_classes['Airplane']] 返回对应
        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]


print('end!')
