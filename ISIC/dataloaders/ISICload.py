from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, ToTensor, ToPILImage
import torch
import glob
from skimage.morphology import square
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple
from transforms.input import GaussianNoise, EnhanceContrast, EnhanceColor
from transforms.target import Opening, ConvexHull, BoundingBox
import numpy as np
import funcy
import matplotlib.pyplot as plt

unloader = ToPILImage()
def imshow(tensor):
	image = unloader(tensor)
	plt.imshow(image)
	plt.pause(0.01)

class ISIC(Dataset):

	def __init__(self, img_dir: str, mask_dir: str,  augmentations: List =None, input_preprocess: Callable =None, target_preprocess: Callable =None,
				 with_targets: bool=True, size: Tuple =(256, 256)):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.size = size
		self.normalize =Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
		self.ids = [file for file in os.listdir(self.img_dir)
					if not file.startswith('.')]
		self.ids.sort(key=lambda x: int(x.split('.')[0][5:]))
		self.resize = Resize(size=self.size)

		self.to_tensor = ToTensor()
		self.input_preprocess = input_preprocess
		self.target_preprocess = target_preprocess
		if augmentations:
			augmentations = [lambda x: x] + augmentations
		else:
			augmentations = [lambda x: x ]

		# self.data = [(idx, augmentation) for augmentation in augmentations for idx in self.ids]
	@staticmethod
	def _load_input_image(fpath: str):
		img = Image.open(fpath).convert("RGB")
		return img

	@staticmethod
	def _load_target_image(fpath: str):
		img = Image.open(fpath).convert("L")
		return img


	def __len__(self):
		# print(self.data)

		return len(self.ids) # 200 * 3

	def __getitem__(self, i):
		# idx, augmentation = self.data[i]
		# print(idx, augmentation)
		idx= self.ids[i]
		# augmentation = self.data[i]
		fullPathName = os.path.join(self.img_dir, idx)
		fullPathName = fullPathName.replace('\\', '/')
		img = self._load_input_image(fullPathName)
		img = self.resize(img)

		if self.input_preprocess is not None:
			img = self.input_preprocess(img)

		# img = augmentation(img)
		img = self.to_tensor(img)
		img = self.normalize(img)

		# get mask
		fullPathName = os.path.join(self.mask_dir, idx)
		fullPathName = fullPathName.replace('\\', '/').split('.')[:-1]
		############################我这里改了 为了适应新的val###########
		# fullPathName = '.'.join(fullPathName)+'_Segmentation.png'
		fullPathName = '.'.join(fullPathName)+'_segmentation_[0-1].png'

		try:
			final_path = glob.glob(fullPathName)
			Mask = self._load_target_image(final_path[0])# Lmode是单通道, 并不妨碍是0-255
			Mask = self.resize(Mask)
			Mask = np.array(Mask)
			Mask = np.where(Mask > 0.5, 1, 0)

		except:
			Mask = np.zeros((256,256))

		# if self.target_preprocess is not None:
		# 	Mask = self.target_preprocess(Mask)
		Mask = self.to_tensor(Mask)
		# print(Mask.size())

		return {'image': img, 'mask': Mask, 'name': idx}


class MultiISIC(Dataset):
	def __init__(self, img_dir: str, mask_dir:str, full_mask_dir=None, augmentations: List=None, input_preprocess: Callable = None, target_preprocess: Callable =None,
				 with_targets: bool=True, size: Tuple =(256, 256), select="all", random_mask = True):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.full_mask_dir = full_mask_dir
		self.size = size
		self.normalize = Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
		self.ids = [file for file in os.listdir(self.img_dir)
					if not file.startswith('.')]
		self.ids.sort(key=lambda x: int(x.split('.')[0][5:]))
		self.resize = Resize(size=self.size)

		self.to_tensor = ToTensor()
		self.input_preprocess = input_preprocess
		self.target_preprocess = target_preprocess
		self.random_mask = random_mask
		if augmentations:
			augmentations = [lambda x: x] + augmentations
		else:
			augmentations = [lambda x: x]
		if select == "all":
			self.selection_method = self._select_all
		else:
			self.selection_method = self._random_selection

		self.data = [(idx, augmentation) for augmentation in augmentations for idx in self.ids]

	@staticmethod
	def _load_input_image(fpath: str):
		img = Image.open(fpath).convert("RGB")
		return img

	@staticmethod
	def _load_target_image(fpath: str):
		img = Image.open(fpath).convert("L")
		return img

	def _select_all(self, targets_list: List[str]):
		target_imgs = []
		# print(targets_list)
		for target_fpath in targets_list:
			target_img = self._load_target_image(target_fpath)
			target_img = self.resize(target_img)
			target_img = np.array(target_img)
			target_img = np.where(target_img > 0.5, 1, 0)
			
			if self.target_preprocess is not None:
				target_img = self.target_preprocess(target_img)
			target_imgs.append(target_img)

		return target_imgs

	def _random_selection(self, targets_list: List[str]):
		target_fpath = np.random.choice(targets_list)

		target_img = self._load_target_image(target_fpath)
		target_img = self.resize(target_img)
		target_img = np.array(target_img)
		target_img = np.where(target_img > 0.5, 1, 0)
		

		if self.target_preprocess is not None:
			target_img = self.target_preprocess(target_img)

		return [target_img]

	def __len__(self):
		# print(len(self.ids))
		return len(self.ids)

	def __getitem__(self, i):
		# idx, augmentation = self.data[i]
		# print(idx, augmentation)
		idx= self.ids[i]
		# augmentation = self.data[i]
		fullPathName = os.path.join(self.img_dir, idx)
		fullPathName = fullPathName.replace('\\', '/')
		img = self._load_input_image(fullPathName)
		img = self.resize(img)

		if self.input_preprocess is not None:
			img = self.input_preprocess(img)

		# img = augmentation(img)
		img = self.to_tensor(img)
		img = self.normalize(img)
		# get mask
		fullPathName = os.path.join(self.mask_dir, idx)
		fullPathName = fullPathName.replace('\\', '/').split('.')[:-1]
		fullPathName = '.'.join(fullPathName)+'_segmentation_[0-1].png'
		final_path = glob.glob(fullPathName)





		if len(final_path)==0:
			idx = idx + 'unlabel'
			Masks = []
			Masks_1 = []
			for i in range(2): # 先看只有2个mask的
				Mask = np.zeros((256, 256))
				Masks.append(Mask)
			Masks = funcy.walk(self.to_tensor, Masks)

		else:
			# print(final_path)
			Masks = self.selection_method(final_path)
			Masks = funcy.walk(self.to_tensor, Masks)


		if self.selection_method == self._random_selection:
			Masks = Masks[0]




		# except:
		# 	Masks = []
		# 	print("==================")
		# 	for i in range(2): # 先看只有2个mask的
		# 		Mask = np.zeros((256, 256))
		# 		Masks.append(self.to_tensor(Mask))
		# 	# Masks = funcy.walk(self.to_tensor, Masks)
		# 	# print("masks",Masks)


		# if self.target_preprocess is not None:
		# 	Mask = self.target_preprocess(Mask)
		# Mask = self.to_tensor(Mask)
		# # print(Mask.size())
		#
		return {'image': img, 'mask': Masks,  'name': idx}


	










if __name__ == '__main__':
	available_conditioning = {
		"original": lambda x: x,
		"opening": Opening(square, 5),
		"convex_hull": funcy.rcompose(Opening(square, 5), ConvexHull()),
		"bounding_box": funcy.rcompose(Opening(square, 5), BoundingBox()),
	}


	# img_dir = '../../dataset/test/images'
	# mask_dir = '../../dataset/test/label'
	# test = ISIC(img_dir=img_dir,mask_dir=mask_dir)
	# print(len(test.ids))
	# print(test.__getitem__(87))
	dir_testImg = '../dataset/train/img150'
	dir_testLabel = '../dataset/train/supervised/multi100'
	augmentations = [
		GaussianNoise(0, 2),
		EnhanceContrast(0.5, 0.1),
		EnhanceColor(0.5, 0.1)
	]
	datasize = (256, 256)
	data = MultiISIC(dir_testImg, dir_testLabel, size=datasize, augmentations=augmentations,target_preprocess=train_preprocess_fn, random_mask=True)
	# print(data.__getitem__(0))

	val_loader = DataLoader(data, batch_sampler=batch_sampler,
									  num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
	for i in val_loader:
		print(i["name"])
	# 	# print(i['mask'].max())
	# 	# print(i['name'])
	# 	# imshow(i)
	# 	print("----")
	print(len(val_loader))

class PH2(Dataset):

	def __init__(self, img_dir: str, mask_dir: str,  augmentations: List =None, input_preprocess: Callable =None, target_preprocess: Callable =None,
				 with_targets: bool=True, size: Tuple =(256, 256)):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.size = size
		self.normalize =Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
		self.ids = [file for file in os.listdir(self.img_dir)
					if not file.startswith('.')]
		self.ids.sort(key=lambda x: int(x.split('.')[0][5:]))
		self.resize = Resize(size=self.size)

		self.to_tensor = ToTensor()
		self.input_preprocess = input_preprocess
		self.target_preprocess = target_preprocess
		if augmentations:
			augmentations = [lambda x: x] + augmentations
		else:
			augmentations = [lambda x: x ]

		# self.data = [(idx, augmentation) for augmentation in augmentations for idx in self.ids]
	@staticmethod
	def _load_input_image(fpath: str):
		img = Image.open(fpath).convert("RGB")
		return img

	@staticmethod
	def _load_target_image(fpath: str):
		img = Image.open(fpath).convert("L")
		return img


	def __len__(self):
		# print(self.data)

		return len(self.ids) # 200 * 3

	def __getitem__(self, i):
		# idx, augmentation = self.data[i]
		# print(idx, augmentation)
		idx= self.ids[i]
		# augmentation = self.data[i]
		fullPathName = os.path.join(self.img_dir, idx)
		fullPathName = fullPathName.replace('\\', '/')
		img = self._load_input_image(fullPathName)
		img = self.resize(img)

		if self.input_preprocess is not None:
			img = self.input_preprocess(img)

		# img = augmentation(img)
		img = self.to_tensor(img)
		img = self.normalize(img)

		# get mask
		fullPathName = os.path.join(self.mask_dir, idx)
		fullPathName = fullPathName.replace('\\', '/').split('.')[:-1]
		############################我这里改了 为了适应新的val###########
		# fullPathName = '.'.join(fullPathName)+'_Segmentation.png'
		fullPathName = '.'.join(fullPathName)+'_lesion.bmp'

		try:
			final_path = glob.glob(fullPathName)
			Mask = self._load_target_image(final_path[0])# Lmode是单通道, 并不妨碍是0-255
			Mask = self.resize(Mask)
			Mask = np.array(Mask)
			Mask = np.where(Mask > 0.5, 1, 0)

		except:
			Mask = np.zeros((256,256))

		# if self.target_preprocess is not None:
		# 	Mask = self.target_preprocess(Mask)
		Mask = self.to_tensor(Mask)
		# print(Mask.size())

		return {'image': img, 'mask': Mask, 'name': idx}