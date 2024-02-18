import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.DD import UnFNet_singal
from dataloaders.ISICload import ISIC, PH2
from torch.nn import functional as F
import argparse
from same_function import available_conditioning,evaluate_jaccard

# def calculate_hd95(pred, gt):
#     batch = pred.shape[0]
#     hd = 0
#     for i in range(batch):
#         if len(np.unique(pred[i, ...])) == 1 and len(np.unique(gt[i, ...])) == 1:
#             hd += 0
#         elif len(np.unique(pred[i, ...])) == 1 or len(np.unique(gt[i, ...])) == 1:
#             hd += 373.13
#         else:
#             hd += metric.binary.hd95(pred[i, ...], gt[i, ...])
#     return hd


def create_model(ema, device):
    # Network definition
    # =======Net========
    # 暂时为False 没有dropout
    model = UnFNet_singal(3, 2, device, l_rate=0.0001, pretrained=False, has_dropout=ema)
    if ema:
        for param in model.parameters():
            param.detach_()  # TODO:反向传播截断
    return model


def test_model(phase, model, dataloader, device):
	
	progress_bar = tqdm(dataloader)
	test = phase == "test"

	if test:
		model.eval()

	# dices_all = []
	# hd95_all = []
	jacces = []

	progress_bar = tqdm(dataloader, desc="Epoch {}".format(phase))
	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]
		Name = data["name"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = label_batch.to(device, dtype=torch.long)


		with torch.no_grad():
			mask_pred = model(volume_batch)
		x = label_batch.squeeze(dim=1)
		mask_true = F.one_hot(x, 2).permute(0, 3, 1, 2).float().cpu()
		mask_pred_0 = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...])
		jacces.append(jacc)
	# 	dice1 = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
	# 	# hd95 = calculate_hd95(np.array(mask_pred_0[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))

	# 	dices_all.append(dice1)
	# 	jacc.append(dice1 / (2 - dice1))
	# 	hd95_all.append(hd95)

	# 	progress_bar.set_postfix(dice1=np.mean(np.array(dices_all)))
	# hd95_all = list(filter(lambda x: x != 373.13, hd95_all))
	# mean_dice = np.mean(np.array(dices_all))
	# mean_jacc = np.mean(np.array(jacc))
	# mean_hd95 = np.mean(np.array(hd95_all))
	# info = {"dice": mean_dice,"jacc":mean_jacc, "hd95":mean_hd95}
	info = {"jacc":round(np.mean(jacces), 4)}
	return info
	
def test_models(phase, model1, model2, dataloader, device):
	
	progress_bar = tqdm(dataloader)
	test = phase == "test"

	if test:
		model1.eval()
		model2.eval()
	# dices_all = []
	# hd95_all = []
	# jacc = []
	jacces = []
	jacces1 = []
	jacces2 = []

	progress_bar = tqdm(dataloader, desc="Epoch {}".format(phase))
	for data in progress_bar:
		volume_batch, label_batch = data["image"], data["mask"]
		Name = data["name"]

		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = label_batch.to(device, dtype=torch.long)


		with torch.no_grad():
			mask_pred1 = model1(volume_batch)
			mask_pred2 = model2(volume_batch)
		mask_pred = [mask_pred1, mask_pred2]
		mask_pred = torch.mean(torch.stack(mask_pred), dim=0)

		x = label_batch.squeeze(dim=1)
		mask_true = F.one_hot(x, 2).permute(0, 3, 1, 2).float().cpu()

		mask_pred_0 = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...])
		jacces.append(jacc)

		mask_pred_1 = F.one_hot(mask_pred1.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc1 = evaluate_jaccard(mask_pred_1[:, 1:2, ...], mask_true[:, 1:2, ...])
		jacces1.append(jacc1)

		mask_pred_2 = F.one_hot(mask_pred2.argmax(dim=1), 2).permute(0, 3, 1, 2).float().cpu()
		jacc2 = evaluate_jaccard(mask_pred_2[:, 1:2, ...], mask_true[:, 1:2, ...])
		jacces2.append(jacc2)


	# 	dice1 = evaluate_jaccard(mask_pred_0[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
	# 	# hd95 = calculate_hd95(np.array(mask_pred_0[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))

	# 	dices_all.append(dice1)
	# 	jacc.append(dice1 / (2 - dice1))
	# 	hd95_all.append(hd95)

	# 	progress_bar.set_postfix(dice1=np.mean(np.array(dices_all)))
	# hd95_all = list(filter(lambda x: x != 373.13, hd95_all))
	# mean_dice = np.mean(np.array(dices_all))
	# mean_jacc = np.mean(np.array(jacc))
	# mean_hd95 = np.mean(np.array(hd95_all))
	# info = {"dice": mean_dice,"jacc":mean_jacc, "hd95":mean_hd95}

	info = {"jacc":round(np.mean(jacces), 4),'jacc1':round(np.mean(jacces1), 4),"jacc2":round(np.mean(jacces2), 4)}
	return info
	
	
def Test_model(model_path, device):
	testdir = 'dataset/test/images'
	testsegdir = 'dataset/test/label'
	img_size = (256, 256)
	model = create_model(ema=False, device=device)  # TODO:创建 model
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	test_preprocess_fn = available_conditioning['original']
	db_val = ISIC(testdir, testsegdir, size=img_size, augmentations=None,
						target_preprocess=test_preprocess_fn)
	dataloaders = {}
	dataloaders["test"] = DataLoader(db_val, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
	info = test_model("test", model=model, dataloader=dataloaders["test"],  device=device)
	return info

def Test_models(model1_path,model2_path,device):
	testdir = 'dataset/test/img'
	testsegdir = 'dataset/test/mask'
	img_size = (256, 256)

	model1 = create_model(ema=False, device=device)  # TODO:创建 model
	model1.load_state_dict(torch.load(model1_path, map_location=device))
	model1.eval()

	model2 = create_model(ema=False, device=device)  # TODO:创建 model
	model2.load_state_dict(torch.load(model2_path, map_location=device))
	model2.eval()


	test_preprocess_fn = available_conditioning['original']
	db_val = ISIC(testdir, testsegdir, size=img_size, augmentations=None,
						target_preprocess=test_preprocess_fn)
	dataloaders = {}
	dataloaders["test"] = DataLoader(db_val, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
	info = test_models("test", model1=model1,model2=model2,dataloader=dataloaders["test"],  device=device)
	return info

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path1", type=str, default="multi_results/Unet_best_200.pth", help="model_path")
	parser.add_argument("--model_path2", type=str, default="multi_results/Unet_best_200.pth", help="model_path")

	parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
	parser.add_argument("--gpu", type=int, default=0, help="gpu")
	parser.add_argument("--fuse", type=int, default=0, help="0 or 1")

	
	args = parser.parse_args()



	device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
	testdir = 'dataset/test/images'
	testsegdir = 'dataset/test/label'
	img_size = (256, 256)


	test_preprocess_fn = available_conditioning['original']

	db_val = ISIC(testdir, testsegdir, size=img_size, augmentations=None,
						target_preprocess=test_preprocess_fn)

	dataloaders = {}
	dataloaders["test1"] = DataLoader(db_val, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False)

	testdir = 'dataset/test/PH2/images'
	testsegdir = 'dataset/test/PH2/labels'
	db_val = PH2(testdir, testsegdir, size=img_size, augmentations=None,
						target_preprocess=test_preprocess_fn)

	dataloaders["test2"] = DataLoader(db_val, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False)

	model_path1 = args.model_path1
	model_path2 = args.model_path2

	model1 = create_model(ema=False, device=device)  # TODO:创建 model
	model1.load_state_dict(torch.load(model_path1, map_location=device))
	if args.fuse == 0:
		info1 = test_model("test", model=model1, dataloader=dataloaders["test1"], device=device)
#info2 = test_model("test", model=model1, dataloader=dataloaders["test2"], device=device)

	if args.fuse == 1:

		model2 = create_model(ema=False, device=device)  # TODO:创建 model
		model2.load_state_dict(torch.load(model_path2, map_location=device))
		info1= test_models("test", model1=model1, model2=model2, dataloader=dataloaders["test1"], device=device)
#info2 = test_models("test", model1=model1, model2=model2, dataloader=dataloaders["test2"], device=device)

	print(info1)
