import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ISAC_rav_HH_Dataset(Dataset):
    """ISAC AoA estimation dataset."""

    def __init__(self, dataframe, input_column='ChannelEstimate_rav', label='AoA', transform=None, sort_target=False,):
        """
        Arguments:
            dataframe (string): Dataframe of simulation data.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        assert label in ('AoA', 'AoD', 'AoA_AoD', 'ToA', 'ToA_AoA_AoD')
        self.orig_label = label
        if label == 'AoA_AoD':
          self.label = ['AoA', 'AoD']
        elif label == 'ToA_AoA_AoD':
           self.label = ['AoA', 'AoD', 'ToA']
        else:  
          self.label = label
        
        self.input_column = input_column
        self.transform = transform
        self.sort_target = sort_target
        self.metadata = {
            'N_targets': int(self.dataframe.q.astype(int).mean()),
        }

        if label in ['AoA_AoD', 'ToA_AoA_AoD']:
          self.dataframe = self.dataframe[[self.input_column] + self.label]
        else:
          self.dataframe = self.dataframe[[self.input_column, self.label]]

    def __len__(self):
        return  len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.loc[idx]
        input = row[self.input_column]
        if self.orig_label in ['AoA_AoD', 'ToA_AoA_AoD']:
          target = np.concatenate(row[self.label].to_numpy())
        else:
          target = row[self.label]

        if self.sort_target:
          target = np.sort(target)

        sample = {
                  'input': input,
                  'target': np.array([target]) if self.label=='ToA' else target,
                }
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToComplexMLPInputsTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input, target  = sample['input'], sample['target']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        return {
                  'input': torch.from_numpy(input).cfloat(),
                  'target': torch.from_numpy(target),
              }