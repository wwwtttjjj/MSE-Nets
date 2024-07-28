import os
import numpy as np
import torch
import random
from tqdm import tqdm
from networks.DD import UnFNet_singal

from utils import ramps
import funcy
from skimage.morphology import square
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import functional as F
import random
from same_function import get_data,get_args,evaluate_jaccard,save_results
from test_sig import Test_models


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# ex = Experiment()
fs_observer = os.path.join(BASE_PATH, "multi_results")  #
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)
loss_fn = CrossEntropyLoss(ignore_index=2)

args = get_args()
lu = args.label_unlabel


save_best1_name = "MSE-Nets_best1_" + str(args.label_unlabel) + '.pth'
save_best2_name = "MSE-Nets_best2_" + str(args.label_unlabel) + '.pth'
if not args.UPC:
	save_best1_name = "MSE_nU-Nets_best1_"+ str(args.label_unlabel) + '.pth'
	save_best2_name = "MSE_nU-Nets_best2_"+ str(args.label_unlabel) + '.pth'

if not args.MPC:
	save_best1_name = "MA-Nets_best1_"+ str(args.label_unlabel) + '.pth'
	save_best2_name = "MA-Nets_best2_"+ str(args.label_unlabel) + '.pth'

if not args.MPC and not args.UPC:
	save_best1_name = "M-Nets_best1_"+ str(args.label_unlabel) + '.pth'
	save_best2_name = "M-Nets_best2_"+ str(args.label_unlabel) + '.pth'


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
		
#val_epoch
def val_epoch(phase, epoch, model1, model2, dataloader):
	
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	val = phase == "val"

	if val:
		model1.eval()
		model2.eval()

	jacces1 = []
	jacces2 = []

	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]
		Name = data["name"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred1 = model1(volume_batch) 
			mask_pred2 = model2(volume_batch)

		mask_true_0 = label_batch[0].squeeze(dim=1)
		mask_true_0 = F.one_hot(mask_true_0, 2).permute(0, 3, 1, 2).float().cpu()
		mask_pred_0 = F.one_hot(mask_pred1.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc1 = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true_0[:, 1:2, ...])
		jacces1.append(jacc1)

		mask_true_1 = label_batch[1].squeeze(dim=1)
		mask_true_1 = F.one_hot(mask_true_1, 2).permute(0, 3, 1, 2).float().cpu()
		mask_pred_1 = F.one_hot(mask_pred2.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc2 = evaluate_jaccard(mask_pred_1[:, 1:2, ...], mask_true_1[:, 1:2, ...])
		jacces2.append(jacc2)

		progress_bar.set_postfix(jacc1=np.mean(jacces1),jacc2=np.mean(jacces2))

	mean_jacc = (np.mean(jacces1) + np.mean(jacces2))/ 2

	info = {"jacc1": np.mean(jacces1), "jacc2": np.mean(jacces2)}

	return info

#train_epoch
def train_epoch(phase, epoch, model1, model2, dataloader, loss_fn):
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	training = phase == "train"

	iter_num = 0

	if training:
		model1.train()
		model2.train()

	for data in progress_bar:
		volume_batch, label_batch, name = data["image"], data["mask"], data['name']  # list len = 2   size = [4, 3, 256, 256]
  # list len = 2   size = [4, 3, 256, 256]
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		if isinstance(label_batch, list):
			targets = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		else:
			targets = label_batch.to(device, dtype=torch.long)

		targets_1 = targets[0][:args.labeled_bs].squeeze(dim=1)
		targets_2 = targets[1][:args.labeled_bs].squeeze(dim=1)

		outputs1 = model1(volume_batch)
		outputs_soft1 = torch.sigmoid(outputs1)
		
		outputs2 = model2(volume_batch)
		outputs_soft2 = torch.sigmoid(outputs2)
		

		pseudo_supervision1 = 0
		pseudo_supervision2 = 0
		value0 = torch.tensor(0).to(device, dtype = torch.long)
		value1 = torch.tensor(1).to(device, dtype = torch.long)
		value2 = torch.tensor(2).to(device, dtype = torch.long)
		if args.MA:	
			binary_map = torch.where(targets_1 == targets_2, value1, value0)
			targets_1 = torch.where(targets_1 == targets_2, targets_1, value2)
			targets_2 = torch.where(targets_1 == targets_2, targets_2, value2)

		if epoch >= 10 and args.MPC:
				labeled_outputs1 = outputs_soft1[:args.labeled_bs].detach()
				labeled_outputs1 = (labeled_outputs1.argmax(dim=1)).long()

				labeled_outputs2 = outputs_soft2[:args.labeled_bs].detach()
				labeled_outputs2 = (labeled_outputs2.argmax(dim=1)).long()
				correct_map = torch.where(labeled_outputs1 == labeled_outputs2, value1, value0).to(device, dtype = torch.long)
				if args.MA:
					c1_b0 = torch.logical_and(correct_map == 1, binary_map == 0).to(device, dtype = torch.long)
					targets_1 = torch.where(c1_b0 == 1, labeled_outputs1, targets_1)
					targets_2 = torch.where(c1_b0 == 1, labeled_outputs2, targets_2)
				else:
					c1_b0 = correct_map
					targets_1 = torch.where(c1_b0 == 1, labeled_outputs1, value2)
					targets_2 = torch.where(c1_b0 == 1, labeled_outputs2, value2)

		if args.UPC:
			pseudo_outputs1 = outputs_soft1[args.labeled_bs:].detach()
			pseudo_outputs1 = (pseudo_outputs1.argmax(dim=1)).long()

			pseudo_outputs2 = outputs_soft2[args.labeled_bs:].detach()
			pseudo_outputs2 = (pseudo_outputs2.argmax(dim=1)).long()
			pseudo_supervision1 = torch.mean(loss_fn(outputs1[args.labeled_bs:], pseudo_outputs2))
			pseudo_supervision2 = torch.mean(loss_fn(outputs2[args.labeled_bs:], pseudo_outputs1))

		loss1 = torch.mean(loss_fn(outputs1[:args.labeled_bs], targets_1))
		loss2 = torch.mean(loss_fn(outputs2[:args.labeled_bs], targets_2))


		consistency_weight = get_current_consistency_weight(iter_num // 150)

		loss = (loss1 + loss2) + consistency_weight * (pseudo_supervision1 + pseudo_supervision2)

		model1.zero_grad()
		model2.zero_grad()
		loss.backward()
		model1.optimize()
		model2.optimize()
		iter_num = iter_num + 1
		progress_bar.set_postfix(loss_unet=loss.item())
		if iter_num % 2000 == 0:
			model1.update_lr()
			model2.update_lr()
	mean_loss = loss

	outputs1 = (outputs1.argmax(dim=1)).float()
	outputs2 = (outputs2.argmax(dim=1)).float()

	info = {"loss": mean_loss,
			}
	return info

def main(args, device):
	base_lr = args.base_lr
	patience = args.patience

	def create_model(ema=False):
		model1 = UnFNet_singal(3, 2, device, l_rate=base_lr, pretrained=True, has_dropout=ema)
		model2 = UnFNet_singal(3, 2, device, l_rate=base_lr, pretrained=True, has_dropout=ema)

		return model1,model2

	model1, model2 = create_model()

	best_model_path1 = os.path.join(fs_observer, save_best1_name)
	best_model_path2 = os.path.join(fs_observer, save_best2_name)

	dataloaders, total_slices = get_data(args)
	info = {}
	epochs = range(0, args.max_iterations // total_slices + 1)
	best_jacc1 = 0
	best_jacc2 = 0

	epochs_since_best = 0


	for epoch in epochs:
		info["train"] = train_epoch("train", epoch, model1=model1, model2=model2, dataloader=dataloaders["train"],
									loss_fn=loss_fn)
		

		info["validation"] = val_epoch("val", epoch, model1=model1, model2=model2, dataloader=dataloaders["validation"])
		#
		if info["validation"]["jacc1"] > best_jacc1:
			best_jacc1= info["validation"]["jacc1"]
			torch.save(model1.state_dict(), best_model_path1)

			epochs_since_best = 0

		if info["validation"]["jacc2"] > best_jacc2:
			best_jacc2= info["validation"]["jacc2"]
			torch.save(model2.state_dict(), best_model_path2)

			epochs_since_best = 0
		else:
			epochs_since_best += 1
		if epochs_since_best > patience:  # 最大容忍不涨区间
			break
	info = Test_models(model1_path= best_model_path1, model2_path = best_model_path2, device=device)

	if not args.UPC and not args.MPC:
		save_results(path = "MSE_M-Nets.txt", info = info, args = args)
	elif not args.UPC:
		save_results(path = "MSE_nU-Nets.txt", info = info, args = args)
	elif not args.MPC:
		save_results(path = "MSE_UPC-Nets.txt", info = info, args = args)
	else:
		save_results(path = "MSE-Nets.txt", info = info, args = args)



if __name__ == '__main__':
	if not args.deterministic:
		cudnn.benchmark = True
		cudnn.deterministic = False
	else:
		cudnn.benchmark = False
		cudnn.deterministic = True

	# random.seed(args.seed)
	# np.random.seed(args.seed)
	# torch.manual_seed(args.seed)
	# torch.cuda.manual_seed(args.seed)
	main(args, device)








