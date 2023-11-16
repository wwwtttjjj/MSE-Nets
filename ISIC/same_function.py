import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from dataloaders.dataset import TwoStreamBatchSampler
from dataloaders.ISICload import MultiISIC
from transforms.input import GaussianNoise, EnhanceContrast, EnhanceColor
from transforms.target import Opening, ConvexHull, BoundingBox
import funcy
from skimage.morphology import square
from torch.nn import functional as F
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_bs", type=int, default=1, help='labeled_batch_size')
    parser.add_argument("--max_iterations", type=int, default=15000,
                        help="maxiumn epoch to train")
    ######################################################
    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    ###################################################
    parser.add_argument("--img_size", type=tuple, default=(256, 256), help="fixed size for img&label")
    parser.add_argument("--imgdir", type=str, default='dataset/train/img', help="path of img")
    # parser.add_argument("--labeldir", type=str, default='dataset/train/supervised/multi30', help="path of label")
    parser.add_argument("--labeldir", type=str, default='dataset/train/mask', help="path of label")
    parser.add_argument("--valdir", type=str, default='dataset/val/img', help="path of validation img")
    parser.add_argument("--valsegdir", type=str, default='dataset/val/mask', help="path of validation label")
    parser.add_argument('--deterministic', type=int, default=0,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='segmentation network learning rate')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument("--patience", type=int, default=30, help="最大容忍不变epoch")
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--label_unlabel', type=str, default='30-170', help='30-170,50-150,70-130,100-100, 100-400, 200-300')
    parser.add_argument('--tar', type=int, default=0, help='0 or 1')
    parser.add_argument('--UPC', type=int, default=1, help='UPC or not')
    parser.add_argument('--MA', type=int, default=1, help='MA or not')
    parser.add_argument('--MPC', type=int, default=1, help='MPC or not')
    parser.add_argument('--ict_alpha', type=int, default=0.2,help='ict_alpha')
	
    args = parser.parse_args()
    return args

def evaluate_jaccard(outputs, targets):
    eps = 1e-15

    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()

    jaccard = (intersection + eps) / (union - intersection + eps)

    return jaccard

#data deal
augmentations = [
	GaussianNoise(0, 2),
	EnhanceContrast(0.5, 0.1),
	EnhanceColor(0.5, 0.1)
]
available_conditioning = {
	"original": lambda x: x,
	"opening": Opening(square, 5),
	"convex_hull": funcy.rcompose(Opening(square, 5), ConvexHull()),
	"bounding_box": funcy.rcompose(Opening(square, 5), BoundingBox()),
}
def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

#val_epoch
def val_epoch(phase, epoch, model, dataloader,args,device):
	
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	val = phase == "val"

	if val:
		model.eval()

	jacces = []

	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]
		Name = data["name"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = model(volume_batch)  # list (len=2)
		mask_true = label_batch[args.tar].squeeze(dim=1)
		mask_true_0 = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float().cpu()
		mask_pred_0 = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true_0[:, 1:2, ...])


		jacces.append(jacc)


		progress_bar.set_postfix(jacces=np.mean(jacces))

	mean_jacces = np.mean(jacces)
	info = {"jacc": mean_jacces}

	return info
def get_data(args):
	labeled = {'30-170':[20, 149, 160],'50-150':[40, 149, 160],'70-130':[60, 149, 160],'100-100':[90, 149, 160],'100-400':[90, 149, 160],'200-300':[199,198,200]}
	train_preprocess_fn = available_conditioning["original"]
	val_preprocess_fn = available_conditioning['original']
	# 载入数据
	db_train = MultiISIC(args.imgdir, args.labeldir, size=args.img_size, augmentations=None,
						 target_preprocess=train_preprocess_fn, select="all")
	# print(db_train.ids)
	db_val = MultiISIC(args.valdir, args.valsegdir, size=args.img_size, augmentations=None,
					   target_preprocess=val_preprocess_fn, select="all")
	total_slices = len(db_train)
	# print(total_slices)
    
	labeled_slices = labeled[args.label_unlabel]
	labeled_idx = list(range(0, labeled_slices[0])) + list(range(labeled_slices[1]+1, labeled_slices[2]))
	unlabeled_idx = list(set(list(range(0, total_slices))) - set(labeled_idx))
	# print(len(unlabeled_idx))
	# print(unlabeled_idx)
	batch_sampler = TwoStreamBatchSampler(
		labeled_idx, unlabeled_idx, args.batch_size, args.batch_size - args.labeled_bs)
	dataloaders = {}
	dataloaders["train"] = DataLoader(db_train, batch_sampler=batch_sampler,
									  num_workers=0, pin_memory=True)
	dataloaders["validation"] = DataLoader(db_val, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
	return dataloaders,total_slices

def save_results(path, info, args):
	if "MSE" in path:
		with open('results_txt/' + path, 'a') as file:
			for key,v in info.items():
				file.write(args.label_unlabel + ':{},{:.4f} '.format(key, v))
				print(key,v)
			file.write('\n')
	else:
		with open('results_txt/' + path, 'a') as file:
			for key,v in info.items():
				file.write(args.label_unlabel + '_' + str(args.tar) + ':{},{:.4f} '.format(key, v))
				print(key,v)
			file.write('\n')