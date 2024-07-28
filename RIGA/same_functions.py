import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
from dataloaders.dataset import TwoStreamBatchSampler
from utils.dataset_loader_cvpr import MyData
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from networks.DD import UnFNet_singal
import dice_score
import torch.nn.functional as F
from dataloaders.dataset import TwoStreamBatchSampler
import funcy
import torch.backends.cudnn as cudnn
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
import os
from utils import ramps
import random

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
fs_observer = os.path.join(BASE_PATH, "24520results")
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)


parameters = dict(
        max_iteration=40000,
        spshot=30,
        nclass=2,
        batch_size=8,
        sshow=655,
        phase="train",  # train or test
        param=False,  # Loading checkpoint
        dataset="Magrabia",  # test or val (dataset)
        snap_num=20,  # Snapshot Number
        gpu_ids='0',  # CUDA_VISIBLE_DEVICES
)

def ave(label_batch, tar):
	if tar <= 6:
		return label_batch[tar]
	if tar == 7:
		most_frequent_values, _ = torch.mode(torch.stack(label_batch[:6]), dim=0)
		return most_frequent_values
	if tar == 8:
		return label_batch[random.randint(0,5)]
def create_model_all(args, ema=False):
    model_all = [UnFNet_singal(3, 3, args.device, l_rate=args.base_lr, pretrained=True, has_dropout=ema) for i in range(args.data_num)]
    return model_all
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_bs", type=int, default=1, help='labeled_batch_size')
    parser.add_argument('--gpu', default=parameters["gpu_ids"], type=str, help='gpu device ids')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--arch', default='resnet34', type=str, help='backbone model name')
    parser.add_argument('--batch_size', default=parameters["batch_size"], type=int, help='batch size for train')
    parser.add_argument('--phase', default=parameters["phase"], type=str, help='train or test')
    parser.add_argument('--param', default=parameters["param"], type=str, help='path to pre-trained parameters')

    parser.add_argument('--train_dataroot', default='./DiscRegion', type=str, help='path to train data')
    parser.add_argument('--test_dataroot', default='./DiscRegion', type=str, help='path to test or val data')
    parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
    parser.add_argument('--val_root', default='./Out/val', type=str, help='directory to save run log')
    parser.add_argument('--log_root', default='./Out/log', type=str, help='directory to save run log')
    parser.add_argument('--snapshot_root', default='./Out/snapshot', type=str, help='path to checkpoint or snapshot')
    parser.add_argument('--output_root', default='./Out/results', type=str, help='path to saliency map')
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument('--seed_init', type=int, default=1337, help='random seed')
    parser.add_argument('--label_unlabel', type=str, default='CP_70_685', help='GPU to use')
    parser.add_argument("--max_iterations", type=int, default=parameters["max_iteration"], help="maxiumn epoch to train")
    #############
    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')
    parser.add_argument('--base_lr', type=float, default=0.00005, help='segmentation network learning rate')
    parser.add_argument('--tar', type=int, default=0, help='0-6, 6 repersent  STAPLE, 7 repersent avergae, 8 respersent random')
    parser.add_argument('--data_num', type=int, default=6, help='2-6')

    parser.add_argument('--aver', type=int, default=0, help='0 or 1')
    parser.add_argument('--random', type=int, default=0, help='0 or 1')
    #UAMT
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
    
    #ICT 专属
    parser.add_argument('--ict_alpha', type=float, default=0.2, help='ict_alpha')
    
    parser.add_argument('--ablation', type=int, default=0, help='whether style of ablation')
    #BCP
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    
    
    args = parser.parse_args()
    
    loss_fn = CrossEntropyLoss()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.loss_fn = loss_fn
    args.fs_observer = fs_observer
    
    if args.aver:
        args.tar = 7
    if args.random:
        args.tar = 8
    return args

def worker_init_fn(worker_id, args):
	random.seed(args.seed + worker_id)
def create_model(args, ema=False):
    # Network definition
    # =======Net========
    # 暂时为False 没有dropout
    model = UnFNet_singal(3, 3, args.device, l_rate=args.base_lr, pretrained=True, has_dropout=ema)
    if ema:
        for param in model.parameters():
            param.detach_()  # TODO:反向传播截断
    return model

def data_loader(args):
    	#load data
    train_sub = MyData(args.train_dataroot, DF=['BinRushed', 'MESSIDOR'], transform=True)
    val_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    total_slices = len(train_sub)
    # 一共655个，label  70
    labeled_slices = [20, 100, 120, 600, 630]
    labeled_idx = list(range(0, labeled_slices[0])) + list(range(labeled_slices[1], labeled_slices[2])) + \
                                list(range(labeled_slices[3], labeled_slices[4]))
    unlabeled_idx = list(set(list(range(0, total_slices))) - set(labeled_idx))

    batch_sampler = TwoStreamBatchSampler(
            labeled_idx, unlabeled_idx, args.batch_size, args.batch_size - args.labeled_bs)
    train_loader = DataLoader(train_sub, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    return train_loader, val_loader
 

def val_epoch(args, phase, epoch, model, dataloader):
	
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	val = phase == "val"

	if val:
		model.eval()

	disc_all = []
	cup_all = []

	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]

		volume_batch = volume_batch.to(args.device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(args.device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = model(volume_batch)
		mask_true = ave(label_batch, args.tar)
		
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
		mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
		
		dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
		dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)

		disc_all.append(dice_disc.item())
		cup_all.append(dice_cup.item())

		progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))

	final_disc = np.mean(disc_all)
	final_cup =  np.mean(cup_all)

	info = {"final_disc": final_disc, "final_cup":final_cup, "mean_dice": (final_disc + final_cup) / 2}

	return info

def get_current_consistency_weight(epoch, args):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def val_all(args, phase, epoch, model, dataloader):
	
    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    val = phase == "val"
    if val:
        funcy.walk(lambda model:model.eval(), model)
    disc_all = [[] for i in range(3)]
    cup_all = [[] for i in range(3)]

    for data in progress_bar:

        volume_batch, label_batch = data["image"], data["mask"]

        volume_batch = volume_batch.to(args.device, dtype=torch.float32)
        label_batch = funcy.walk(lambda target: target.to(args.device, dtype=torch.long), label_batch)

        with torch.no_grad():
            mask_pred = [model[i](volume_batch) for i in range(args.data_num)]
        mask_pred = torch.mean(torch.stack(mask_pred), dim=0)
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()

        for tar in range(6, 9):
            mask_true = ave(label_batch, tar)
            mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
            dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
            dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
            disc_all[tar - 6].append(dice_disc.item())
            cup_all[tar - 6].append(dice_cup.item())
        

        progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))

    final_disc = funcy.walk(lambda disc:round(np.mean(disc), 5), disc_all)
    final_cup =  funcy.walk(lambda cup:round(np.mean(cup), 5), cup_all)


    info = {"final_disc": final_disc, "final_cup":final_cup}

    return info

def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
