"""
This script plots the performance comparison of MLP with the 2D Estimation algorithm 
for both AoA and AoD angles for settings with two targets per peak
Simulation data are obtained by running the Compare2TargetsPeak.m MATLAB script and saved as mat files.
2D Estimator results and CRB bounds are also generated using the MATLAB script
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
from functools import partial 

from datasets import create_dataset
from models import HH2ComplexMLP
from loaders import ISAC_rav_HH_Dataset, ToComplexMLPInputsTensor
from trainer import train, compare_models

import matplotlib
matplotlib.rcParams['font.family'] = "DejaVu Sans"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

style_label = 'seaborn-v0_8-deep'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Creating the Dataset 
train_snr_values =  [5, 15, 20, 25, 30, 40]
for snr_val in train_snr_values:
    data_path = f'./data/Bistatic_data_with_ToA_2Targets_1peak_AoA_AoD/Bistatic_data_with_ToA_2Targets_1peak_{snr_val:d}_dB'
    csv_path = f'./csv_data/Bistatic_data_with_ToA_2Targets_1peak_{snr_val:d}_dB_df.csv'

    if ((not os.path.exists(csv_path)) and (os.path.exists(data_path))):
        print(f'SNR: {snr_val}\n')
        df_sim = create_dataset(data_path=data_path, save_dir=csv_path,
                                columns_to_include=['TimeChannelEstimate'],
                                include_2D_3D=True)

# Creating Training DataFrame
all_sim_df = pd.DataFrame()
snr_bar = tqdm.tqdm(train_snr_values)
snr_bar.set_description("Creating Training DataFrame")
for snr_value in snr_bar:
   path_csv = f'./csv_data/Bistatic_data_with_ToA_2Targets_1peak_{snr_value:d}_dB_df.csv'
   if os.path.exists(path_csv):
      tmp_df = pd.read_pickle(path_csv)
      tmp_df['TimeChannelEstimate_rav'] = tmp_df.TimeChannelEstimate.apply(np.ravel)
      all_sim_df = pd.concat((all_sim_df,tmp_df), ignore_index=True)


# Creating Training DataLoaders 
idx_train, idx_test = train_test_split(range(all_sim_df.shape[0]),test_size=0.02, random_state=2022)

all_sim_df['SET'] = 'TRAIN'
all_sim_df.loc[idx_test, ['SET']] = 'TEST'

df_train = all_sim_df[all_sim_df.SET == 'TRAIN'].copy(deep=True)
df_test = all_sim_df[all_sim_df.SET == 'TEST'].copy(deep=True)

LABEL = 'AoA_AoD'

isac_train_dataset = ISAC_rav_HH_Dataset(dataframe=df_train,
                                       input_column='TimeChannelEstimate_rav',
                                       label=LABEL, 
                                       transform=transforms.Compose([ToComplexMLPInputsTensor()]),
                                       sort_target=False,
                                       )
isac_train_dataloader = DataLoader(isac_train_dataset, batch_size=128, shuffle=True, num_workers=0)

# Test dataset
isac_test_dataset = ISAC_rav_HH_Dataset(dataframe=df_test,
                                       input_column='TimeChannelEstimate_rav',
                                       label=LABEL,
                                       transform=transforms.Compose([ToComplexMLPInputsTensor()]),
                                       sort_target=False,
                                       )
isac_test_dataloader = DataLoader(isac_test_dataset, batch_size=128, shuffle=True, num_workers=0)

# Training the MLP architecture if model is not fitted yet

model_name = f'Bistatic_data_with_ToA_2Targets_1peak_mixeddB'
PATH = f'./models/{model_name}.pt'
if not os.path.exists(PATH):
  N_OUTPUTS = 4
  N_EPOCHS = 350
  LR = 2e-4

  complex_model = HH2ComplexMLP(input_size=80, device=device, n_outputs=N_OUTPUTS)
  # Train the model
  complex_model = complex_model.to(device)

  net, history_test = train(
                          complex_model ,
                          isac_train_dataloader,
                          isac_test_dataloader,
                          epochs=N_EPOCHS,
                          test_metric='rad_mse',
                          lr=LR,
                          plot_save_dir=f'./results/{model_name}.png',
                          save_directory=f'./models/{model_name}',
                          n_epochs_save_model=280,
                          multi_label=True,
                          multi_target=True,
                          use_schedulers=True,
                          n_outputs=N_OUTPUTS,
                          milestones=[330],
                      )
  # Save the model
  print('Saving the model')
  torch.save(net.state_dict(), PATH)

# Creating Test set 
data_path = f'./data/Bistatic_data_with_ToA_2Targets_1peak_AoA_AoD_test_withCRB'
csv_path = f'./csv_data/Bistatic_data_with_ToA_2Targets_1peak_AoA_AoD_test_df_withCRB.csv'
if ((not os.path.exists(csv_path)) and (os.path.exists(data_path))):

    df_sim = create_dataset(data_path=data_path,
                            save_dir=csv_path,
                            columns_to_include=['TimeChannelEstimate'],
                            include_2D_3D=True, 
                            include_CRB=True)

all_snr_df = pd.read_pickle('./csv_data/Bistatic_data_with_ToA_2Targets_1peak_AoA_AoD_test_df_withCRB.csv')
all_snr_df['TimeChannelEstimate_rav'] = all_snr_df.TimeChannelEstimate.apply(np.ravel)
all_snr_df = all_snr_df.explode(['snr'])

LABEL = 'AoA'
N_OUTPUTS = 2

dict_models = {}
PATH = './models/Bistatic_data_with_ToA_2Targets_1peak_mixeddB.pt'
joint_all_db_model = HH2ComplexMLP(input_size=80, device=device, n_outputs=2*N_OUTPUTS)
joint_all_db_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
joint_all_db_model.eval()

dict_models['MLP_AlldB'] = (joint_all_db_model, partial(ISAC_rav_HH_Dataset, input_column='TimeChannelEstimate_rav',
                                                        label='AoA_AoD', transform=transforms.Compose([ToComplexMLPInputsTensor()])), 
                            'single-input', 'multi_label_target')

df_AoA_mean = compare_models(all_snr_df, list_models=dict_models, label=LABEL, include_2D=True, include_3D=False, return_df=True, show_ci=False, 
                             normalized_mse=True, multi_target=True, aggregate_function=np.median, title=f'Prediction of {LABEL} for one peak value with 2 Targets',
                               legend_half=False, include_CRB=True, plot=False)

LABEL = 'AoD'
N_OUTPUTS = 2

dict_models = {}
PATH = './models/Bistatic_data_with_ToA_2Targets_1peak_mixeddB.pt'
joint_all_db_model = HH2ComplexMLP(input_size=80, device=device, n_outputs=2*N_OUTPUTS)
joint_all_db_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
joint_all_db_model.eval()

dict_models['MLP_AlldB'] = (joint_all_db_model, partial(ISAC_rav_HH_Dataset, input_column='TimeChannelEstimate_rav', 
                                                        label='AoA_AoD', transform=transforms.Compose([ToComplexMLPInputsTensor()])),
                            'single-input', 'multi_label_target')

df_AoD_mean = compare_models(all_snr_df, list_models=dict_models, label=LABEL, include_2D=True, include_3D=False, return_df=True, show_ci=False, 
                             normalized_mse=True, multi_target=True, aggregate_function=np.median, title=f'Prediction of {LABEL} for one peak value with 2 Targets', 
                             legend_half=False, include_CRB=True, plot=False)

df_AoA_mean['Angle'] = 'AoA'
df_AoD_mean['Angle'] = 'AoD'

df_results_2tar = pd.concat((df_AoA_mean, df_AoD_mean), ignore_index=True)
df_plot_res = df_results_2tar.query('SNR <= 25')

# Plotting the results
df_plot_res = df_plot_res[df_plot_res.SNR.isin(np.arange(-5, 30, step=3))]
with plt.style.context(style_label, {'grid.linestyle': '---'}):
  fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
  sns.lineplot(data=df_plot_res, x="SNR", y="MSE", hue="Model", ax=ax, style='Angle', markersize=16, markers=True, dashes=True, errorbar=None, estimator=np.mean,palette=['blue', 'red', 'dodgerblue'], linewidth=3)
  ax.set_yscale('log')
  ax.set_xlabel(r'$SNR [dB]$', fontsize=24)
  ax.set_ylabel(r'$MSE/rad^{2}$', fontsize=24)
  ax.set_xlim([df_plot_res.SNR.min(), df_plot_res.SNR.max()])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), fancybox=True, shadow=True,ncols=2, fontsize=22)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.4)
  ax.grid(which='major', alpha=0.9)
  ax.tick_params(axis='both', which='major', labelsize=24)
  ax.tick_params(axis='both', which='minor', labelsize=24)

  plt.tight_layout()
  plt.show()