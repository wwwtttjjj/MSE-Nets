import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
from dataloaders.dataset import TwoStreamBatchSampler
from utils.dataset_loader_cvpr import MyData
import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from networks.DD import UnFNet_singal
from utils import dice_score
import torch.nn.functional as F
from dataloaders.dataset import TwoStreamBatchSampler
import funcy
import torch.backends.cudnn as cudnn
# import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from same_functions import ave
import os
from utils import ramps
import random
os.environ["WANDB_MODE"] = "dryrun"
#random generate int (0,6)
def generate_list():
    while True:
        # 生成包含0到5的列表
        lst = list(range(6))
        # 打乱列表的顺序
        random.shuffle(lst)
        
        # 检查每个元素的索引是否与其值相同
        if all(i != lst[i] for i in range(len(lst))):
            return lst
        
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
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--val_root', default='./Out/val', type=str, help='directory to save run log')
parser.add_argument('--log_root', default='./Out/log', type=str, help='directory to save run log')
parser.add_argument('--snapshot_root', default='./Out/snapshot', type=str, help='path to checkpoint or snapshot')
parser.add_argument('--output_root', default='./Out/results', type=str, help='path to saliency map')
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--patience", type=int, default=100, help="最大容忍不变epoch")
parser.add_argument('--seed_init', type=int, default=1337, help='random seed')
parser.add_argument('--label_unlabel', type=str, default='MCPLD-70-585', help='GPU to use')
parser.add_argument("--max_iterations", type=int, default=parameters["max_iteration"], help="maxiumn epoch to train")
#############
parser.add_argument('--consistency_type', type=str,
					default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
					default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
					default=200.0, help='consistency_rampup')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--base_lr', type=float, default=0.00005,
					help='segmentation network learning rate')
parser.add_argument('--aver', type=int, default=0, help='0 or 1')
parser.add_argument('--random', type=int, default=0, help='0 or 1')



#############
args = parser.parse_args()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.device = device

args.fs_observer = fs_observer
if args.aver:
	args.tar = 6
if args.random:
	args.tar = 7
 

save_best_name = "MSE-Net_best_70-585_"

# W_name = args.label_unlabel
# experiment = wandb.init(project='Mutil_RIGA_Unet', resume='allow', anonymous='must', name=W_name)
# experiment.config.update(dict(batch_size=args.batch_size, labeled_bs = args.labeled_bs, learning_rate=args.base_lr))

loss_fn = CrossEntropyLoss(ignore_index=3)

def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
		
#val
def cal_dice(mask_pred, mask_true):
	mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
	mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
	dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
	dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
	return dice_disc, dice_cup


def val_epoch(args, phase, epoch, model, dataloader, fuse):
	
	progress_bar = tqdm(dataloader, desc="{}".format(phase))
	l = len(dataloader)
	val = phase == "val"

	if val:
		funcy.walk(lambda model:model.eval(), model)

	disc_all = []
	cup_all = []

	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]

		volume_batch = volume_batch.to(args.device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(args.device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = [model[i](volume_batch) for i in range(fuse)]
		mask_pred = torch.mean(torch.stack(mask_pred), dim=0)
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
		mask_true = ave(label_batch, args.tar)
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
#train
def correct_unlabled_mask_f(pseudo_outputs):
		value3 = torch.tensor(3).to(device, dtype = torch.long)

		correct_unlabled_mask0 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[1] == pseudo_outputs[2],pseudo_outputs[2] == pseudo_outputs[3])
												,pseudo_outputs[3] == pseudo_outputs[4])
												,pseudo_outputs[4] == pseudo_outputs[5])
		target0 = torch.where(correct_unlabled_mask0==1, pseudo_outputs[1], value3)
		correct_unlabled_mask1 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[0] == pseudo_outputs[2],pseudo_outputs[2] == pseudo_outputs[3])
												,pseudo_outputs[3] == pseudo_outputs[4])
												,pseudo_outputs[4] == pseudo_outputs[5])
		target1 = torch.where(correct_unlabled_mask1==1, pseudo_outputs[2], value3)
		
		correct_unlabled_mask2 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[0] == pseudo_outputs[1],pseudo_outputs[1] == pseudo_outputs[3])
												,pseudo_outputs[3] == pseudo_outputs[4])
												,pseudo_outputs[4] == pseudo_outputs[5])
		target2 = torch.where(correct_unlabled_mask2==1, pseudo_outputs[3], value3)
		
		correct_unlabled_mask3 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[0] == pseudo_outputs[1],pseudo_outputs[1] == pseudo_outputs[2])
												,pseudo_outputs[2] == pseudo_outputs[4])
												,pseudo_outputs[4] == pseudo_outputs[5])
		target3 = torch.where(correct_unlabled_mask3==1, pseudo_outputs[4], value3)
		
		correct_unlabled_mask4 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[0] == pseudo_outputs[1],pseudo_outputs[1] == pseudo_outputs[2])
												,pseudo_outputs[2] == pseudo_outputs[3])
												,pseudo_outputs[3] == pseudo_outputs[5])
		target4 = torch.where(correct_unlabled_mask4==1, pseudo_outputs[5], value3)
		
		correct_unlabled_mask5 = torch.logical_and(torch.logical_and(torch.logical_and(pseudo_outputs[0] == pseudo_outputs[1],pseudo_outputs[1] == pseudo_outputs[2])
												,pseudo_outputs[2] == pseudo_outputs[3])
												,pseudo_outputs[3] == pseudo_outputs[4])
		target5 = torch.where(correct_unlabled_mask5==1, pseudo_outputs[0], value3)
		
		target = [target0,target1,target2,target3,target4,target5]

		return target
def train_epoch(phase, epoch, model, dataloader, loss_fn):
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	training = phase == "train"

	iter_num = 0

	if training:
		funcy.walk(lambda model:model.train(), model)


	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]  # list len = 2   size = [4, 3, 256, 256]
  # list len = 2   size = [4, 3, 256, 256]
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		if isinstance(label_batch, list):
			targets = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		else:
			targets = label_batch.to(device, dtype=torch.long)
		targets_all = funcy.walk(lambda out:out[:args.labeled_bs], targets)
		
		outputs = [model[i](volume_batch) for i in range(6)]
		#soft output
		outputs_soft1_all = funcy.walk(lambda out:torch.sigmoid(out), outputs)
		#supervised loss
		
		#consistency loss
		consistency_weight = get_current_consistency_weight(iter_num // 150)

		pseudo_supervision_all = []
		pseudo_outputs = funcy.walk(lambda out:out
			       [args.labeled_bs:].detach().argmax(dim=1).long(), outputs_soft1_all)
		correct_unlabled_mask = correct_unlabled_mask_f(pseudo_outputs)
			
		random_index = generate_list()


		pseudo_supervision_all = [torch.mean(loss_fn(outputs[i][args.labeled_bs:], correct_unlabled_mask[i])) for i in range(6)]
		
		
		super_loss_all = []
		value0 = torch.tensor(0).to(device, dtype = torch.long)
		value1 = torch.tensor(1).to(device, dtype = torch.long)
		value2 = torch.tensor(3).to(device, dtype = torch.long)

		for i in range(6):
			binary_map = torch.where(targets_all[i] == targets_all[random_index[i]], value1, value0)
			targets_1 = torch.where(binary_map==1, targets_all[i], value2)

			if epoch >= 10:
				labeled_outputs1 = outputs_soft1_all[i][:args.labeled_bs].detach()
				labeled_outputs1 = (labeled_outputs1.argmax(dim=1)).long()

				labeled_outputs2 = outputs_soft1_all[random_index[i]][:args.labeled_bs].detach()
				labeled_outputs2 = (labeled_outputs2.argmax(dim=1)).long()

				correct_map = torch.where(labeled_outputs1 == labeled_outputs2, value1, value0).to(device, dtype = torch.long)

				c1_b0 = torch.logical_and(correct_map == 1, binary_map == 0).to(device, dtype = torch.long)
				binary_map = torch.where(c1_b0 == 1, value1, binary_map)
				targets_1 = torch.where(c1_b0 == 1, labeled_outputs1, targets_1)

			l = torch.mean(loss_fn(outputs[i][:args.labeled_bs], targets_1))

			super_loss_all.append(l)

		#total_loss
		loss = 0.5 * torch.sum(torch.stack(super_loss_all)) + consistency_weight * (torch.sum(torch.stack(pseudo_supervision_all)))*0.5
		funcy.walk(lambda model:model.zero_grad(), model)
		loss.backward()
		funcy.walk(lambda model:model.optimize(), model)

		iter_num = iter_num + 1
		progress_bar.set_postfix(loss_unet=loss.item())
		if iter_num % 2000 == 0:
			funcy.walk(lambda model:model.update_lr(), model)
			
	mean_loss = loss

	outputs_image = [(outputs[i].argmax(dim=1)).float() for i in range(6)]

	info = {"loss": mean_loss,
			}
	
	return info

#main
def main(args, device, multask=True):
	batch_size = args.batch_size
	base_lr = args.base_lr
	patience = args.patience
	def create_model(ema=False):
		model1 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model2 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model3 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model4 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model5 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model6 = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=ema)

		return [model1, model2, model3, model4, model5, model6]
	model_all = create_model()
	def worker_init_fn(worker_id):
		random.seed(args.seed + worker_id)

	best_model_path = os.path.join(fs_observer, save_best_name)

	#load data
	train_sub = MyData(args.train_dataroot, DF=['BinRushed', 'MESSIDOR'], transform=True)
	val_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
	val_loader = DataLoader(val_sub, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
	test_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
	test_loader = DataLoader(test_sub, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
	total_slices = len(train_sub)
	# 一共655个，label  70
	labeled_slices = [20, 100, 120, 600, 630]
	labeled_idx = list(range(0, labeled_slices[0])) + list(range(labeled_slices[1], labeled_slices[2])) + \
								list(range(labeled_slices[3], labeled_slices[4]))
	unlabeled_idx = list(set(list(range(0, total_slices))) - set(labeled_idx))

	batch_sampler = TwoStreamBatchSampler(
			labeled_idx, unlabeled_idx, args.batch_size, args.batch_size - args.labeled_bs)
	train_loader = DataLoader(train_sub, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
 
	final_disc = []
	final_cup = []
 

	for epoch in range(0, args.epochs):
		start = time.time()
		info = train_epoch("train", epoch, model=model_all, dataloader=train_loader,loss_fn=loss_fn)
		end = time.time()
		print(start-end)								
		#计算最后五个epcohs的mean-dice和std
		if epoch + 5 >=  args.epochs:
			start = time.time()
			info = val_epoch(args, "val", epoch, model=model_all, dataloader=val_loader, fuse=6)
			end = time.time()
			print(start-end)
			
			final_disc.append(info["final_disc"])
			final_cup.append(info["final_cup"])
   
	mean_disc = round(np.mean(final_disc), 4)
	disc_std = round(np.std(final_disc), 4)

	mean_cup = round(np.mean(final_cup), 4)
	cup_std = round(np.std(final_cup), 4)
	
	# 构造文件名
	filename = os.path.join(args.fs_observer, "MSE-Net" + "_" + str(args.tar) + ".txt")

	# 以追加模式打开文件，并写入数据
	with open(filename, 'a') as f:
		f.write("DISC:"f"{mean_disc}  {disc_std}\n")
		f.write("CUP:"f"{mean_cup}  {cup_std}\n")
  


if __name__ == '__main__':
	if not args.deterministic:
		cudnn.benchmark = True
		cudnn.deterministic = False
	else:
		cudnn.benchmark = False
		cudnn.deterministic = True

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	main(args, device, multask=True)

