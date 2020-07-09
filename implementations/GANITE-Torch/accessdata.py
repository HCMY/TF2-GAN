import pickle
from torch.utils.data import Dataset
import numpy as np



class SimpleDataLoader(Dataset):

	def __init__(self, phase='train'):
		"""
		"""
		with open('./data/jobs/jobs_info.pkl','rb') as f:
			info = pickle.load(f)

		x, t, yf = info[phase]['x'],info[phase]['t'],info[phase]['yf']
		self.x = x
		self.t = t
		self.yf = yf


	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		x = self.x[idx]
		yf = self.yf[idx]
		t = self.t[idx]

		return x, np.asarray([yf]).astype(float), np.asarray([t]).astype(float)