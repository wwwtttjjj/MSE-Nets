import torch
import numpy as np
import torch
from tqdm import tqdm
import dice_score
import torch.nn.functional as F
import funcy
import argparse
from networks.DD import UnFNet_singal
from torch.utils.data import DataLoader
from utils.dataset_loader_cvpr import MyData
from collections import Counter

random_index = np.load('random_array.npy')

def ave(label_batch):
	most_frequent_values, _ = torch.mode(torch.stack(label_batch), dim=0)
	return most_frequent_values
def Test_model(phase, model, dataloader, device):
	
	progress_bar = tqdm(dataloader, desc="{}".format(phase))
	test = phase == "test"

	if test:
		model.eval()

	disc_all = [[] for i in range(8)]
	cup_all = [[] for i in range(8)]

	aver =0 
	r_num = 0
	for data in progress_bar:
		if aver == 6:
			aver = 0 
		volume_batch, label_batch = data["image"], data["mask"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = model(volume_batch)
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
			
		for tar in range(8):
			if tar == 6:
				mask_true = ave(label_batch)
			elif tar == 7:
				mask_true = label_batch[random_index[r_num]]
			else:
				mask_true = label_batch[tar]
			mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
			dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
			dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
			disc_all[tar].append(dice_disc.item())
			cup_all[tar].append(dice_cup.item())
		progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))
		
		aver += 1
		r_num += 1
		
	final_disc = funcy.walk(lambda disc:np.mean(disc), disc_all)
	final_cup =  funcy.walk(lambda cup:np.mean(cup), cup_all)
	

	info = {"final_disc": final_disc, "final_cup":final_cup}
	return info
def create_model(device, ema=False):
	model = UnFNet_singal(3, 3, device, l_rate=0.00005, pretrained=True, has_dropout=ema)
	return model

def Test_models(phase, model, dataloader, device, fuse):
	
	progress_bar = tqdm(dataloader, desc="{}".format(phase))
	l = len(dataloader)
	val = phase == "test"

	if val:
		funcy.walk(lambda model:model.eval(), model)

	disc_all = [[] for i in range(8)]
	cup_all = [[] for i in range(8)]
	aver=0
	r_num = 0
	for data in progress_bar:
		if aver == 6:
			aver = 0
		volume_batch, label_batch = data["image"], data["mask"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = [model[i](volume_batch) for i in range(fuse)]
		mask_pred = torch.mean(torch.stack(mask_pred), dim=0)
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()

		for tar in range(8):
			if tar == 6:
				mask_true = ave(label_batch)
			elif tar == 7:
				mask_true = label_batch[random_index[r_num]]
			else:
				mask_true = label_batch[tar]
			mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
			dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
			dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
			disc_all[tar].append(dice_disc.item())
			cup_all[tar].append(dice_cup.item())
		
		aver += 1
		r_num += 1
		progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))

	final_disc = funcy.walk(lambda disc:np.mean(disc), disc_all)
	final_cup =  funcy.walk(lambda cup:np.mean(cup), cup_all)
	

	info = {"final_disc": final_disc, "final_cup":final_cup}

	return info
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, default="RIGA_results/BCP_best_70_585_0.pth", help="model_path")
	parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
	parser.add_argument("--gpu", type=int, default=0, help="gpu")
	parser.add_argument('--test_dataroot', default='./DiscRegion', type=str, help='path to test or val data')
	parser.add_argument('--fuse', default=0, type=int, help='the num of network')


	args = parser.parse_args()
	device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


	model_path = args.model_path
	model = create_model(device=device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	
	val_sub = MyData(args.test_dataroot, DF=["Magrabia"])
	val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
	if args.fuse:
		model_all = []
		path_a = args.model_path[:-5]
		for i in range(args.fuse):
			path = path_a + str(i) + '.pth'
			model = create_model(device=device)
			model.load_state_dict(torch.load(path, map_location=device))
			model_all.append(model)
		info = Test_models(phase = "test", model = model_all, dataloader = val_loader, device = device, fuse = args.fuse)
		
	else:
		info = Test_model(phase = "test", model = model, dataloader = val_loader, device = device)
	print(info)

