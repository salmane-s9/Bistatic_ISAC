"""
This script plots the performance of the MLP architecture trained at different SNR values
and used for prediction on a range of Test SNRs
Simulation data are obtained by running the SNRAnalysis.m MATLAB script and saved as mat files.
"""

import os
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
from trainer import train_SNR_analysis, snr_analysis_comparison

import matplotlib
matplotlib.rcParams['font.family'] = "DejaVu Sans"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

style_label = 'seaborn-v0_8-deep'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


# Training dataset SNR values
train_snr_values = [5, 10, 15, 20, 25, 30, 40]
for snr_val in train_snr_values:
    data_path = f'./data/Bistatic_data_with_ToA_3targets_new/Bistatic_data_with_ToA_3targets_new_2D_{snr_val:d}_dB'
    csv_path = f'./csv_data/Bistatic_data_with_ToA_3targets_new_2D_{snr_val:d}_dB_df.csv'

    if ((not os.path.exists(csv_path)) and (os.path.exists(data_path))):
        if len(os.listdir(data_path))>=10000:
            print(f'SNR: {snr_val}\n')
            df_sim = create_dataset(data_path=data_path,  save_dir=csv_path,
                                    columns_to_include=['TimeChannelEstimate', 'ChannelIFFT'], include_2D_3D=True)
            

# Training Models on specific SNRs

N_TARGETS = 6
N_EPOCHS = 200
LR = 3e-4
for snr_value in train_snr_values:
    model_name = f'Bistatic_3D_NEW_joint_AoA_AoD_{snr_value:d}bB'
    PATH = f'./models/{model_name}.pt'
    if not f'{model_name}.pt' in os.listdir('./models/'):
        print(f'Trainin on SNR={snr_value} dB\n')
        all_sim_df = pd.read_pickle(f'./csv_data/Bistatic_data_with_ToA_3targets_new_2D_{snr_value:d}_dB_df.csv')
        all_sim_df['TimeChannelEstimate_rav'] = all_sim_df.TimeChannelEstimate.apply(np.ravel)

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

        print(f'Training with Device: {device}')
        complex_model = HH2ComplexMLP(input_size=240, device=device, n_outputs=N_TARGETS)
        # Train the model
        complex_model = complex_model.to(device)

        net, history_test = train_SNR_analysis(
                                complex_model ,
                                isac_train_dataloader,
                                isac_test_dataloader,
                                epochs=N_EPOCHS,
                                test_metric='rad_mse',
                                lr=LR,
                                plot_save_dir=f'./results/{model_name}.png',
                                save_directory=f'./models/{model_name}',
                                n_epochs_save_model=120,
                                multi_label=True,
                                multi_target=True,
                                use_schedulers=False,
                                n_outputs=N_TARGETS,
                            )

        # Save the model
        print('Saving the model')
        torch.save(net.state_dict(), PATH)


# Training on All the SNRs (MLP_AlldB) 
N_TARGETS = 6
N_EPOCHS = 200
LR = 3e-4

# Creating the Dataset
all_sim_df = pd.DataFrame()
snr_bar = tqdm.tqdm(train_snr_values)
snr_bar.set_description("Creating Training DataFrame")
for snr_value in snr_bar:
   tmp_df = pd.read_pickle(f'./csv_data/Bistatic_data_with_ToA_3targets_new_2D_{snr_value:d}_dB_df.csv')
   tmp_df['TimeChannelEstimate_rav'] = tmp_df.TimeChannelEstimate.apply(np.ravel)
   tmp_df = tmp_df.sample(n=10)
   all_sim_df = pd.concat((all_sim_df,tmp_df), ignore_index=True)

# Splitting the training dataset to Train and Validation   
idx_train, idx_test = train_test_split(range(all_sim_df.shape[0]),test_size=0.02, random_state=2022)

all_sim_df['SET'] = 'TRAIN'
all_sim_df.loc[idx_test, ['SET']] = 'TEST'

df_train = all_sim_df[all_sim_df.SET == 'TRAIN'].copy(deep=True)
df_test = all_sim_df[all_sim_df.SET == 'TEST'].copy(deep=True)

# Joint prediction of AoA-AoD
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

print(f'Training with Device: {device}')
complex_model = HH2ComplexMLP(input_size=240, device=device, n_outputs=N_TARGETS)
# Train the model
complex_model = complex_model.to(device)    

PATH = './models/Bistatic_3D_NEW_joint_AoA_AoD_mixedbB.pt'
if not os.path.exists(PATH):
    net, history_test = train_SNR_analysis(
                            complex_model ,
                            isac_train_dataloader,
                            isac_test_dataloader,
                            epochs=N_EPOCHS,
                            test_metric='rad_mse',
                            lr=LR,
                            plot_save_dir='./results/Bistatic_3D_NEW_joint_AoA_AoD_mixedbB.png',
                            save_directory='./models/Bistatic_3D_NEW_joint_AoA_AoD_mixedbB',
                            n_epochs_save_model=120,
                            multi_label=True,
                            multi_target=True,
                            use_schedulers=False,
                            n_outputs=N_TARGETS,
                        )

    # Save the model
    print('Saving the model')
    torch.save(net.state_dict(), PATH)


# Plotting Test results

# Reading Test Data
all_snr_df = pd.read_pickle('./csv_data/Bistatic_data_with_ToA_3targets_new_2D_test_gamma_1_df.csv')
all_snr_df['TimeChannelEstimate_rav'] = all_snr_df.TimeChannelEstimate.apply(np.ravel)
all_snr_df = all_snr_df.explode(['snr'])

LABEL = 'AoA'

dict_models = {}
for snr_value in train_snr_values:
  model_name = f'Bistatic_3D_NEW_joint_AoA_AoD_{snr_value:d}bB'
  PATH = f'./models/{model_name}.pt'
  tmp_model = HH2ComplexMLP(input_size=240, device=device, n_outputs=N_TARGETS)
  tmp_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
  tmp_model.eval()

  dict_models[f'MLP_{snr_value}dB'] = (tmp_model, 
                                       partial(ISAC_rav_HH_Dataset, input_column='TimeChannelEstimate_rav', label='AoA_AoD',
                                               transform=transforms.Compose([ToComplexMLPInputsTensor()])), 
                                       'single-input', 'multi_label_target')

PATH = './models/Bistatic_3D_NEW_joint_AoA_AoD_mixedbB.pt'
joint_all_db_model = HH2ComplexMLP(input_size=240, device=device, n_outputs=N_TARGETS)
joint_all_db_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
joint_all_db_model.eval()
dict_models['MLP_AlldB'] = (joint_all_db_model,
                            partial(ISAC_rav_HH_Dataset,input_column='TimeChannelEstimate_rav', label='AoA_AoD',
                                    transform=transforms.Compose([ToComplexMLPInputsTensor()])),
                            'single-input', 'multi_label_target')


df_AoA_mean = snr_analysis_comparison(all_snr_df, list_models=dict_models, label=LABEL, include_2D=False, include_3D=False, return_df=True,
                                      show_ci=False, normalized_mse=True, multi_target=True, aggregate_function=np.median, plot=False)

mse_all = df_AoA_mean.copy()

mse_all = mse_all[mse_all.SNR.isin(np.arange(-5, 32, step=3))]
with plt.style.context(style_label, {'grid.linestyle': '---'}):
  fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
  sns.lineplot(data=mse_all, x="SNR", y="MSE", hue="Model", ax=ax, style='Model', markersize=16, markers=True, dashes=True, errorbar=None, estimator=np.mean, linewidth=3)

  ax.set_yscale('log')
  ax.set_xlabel(r'$SNR [dB]$', fontsize=24)
  ax.set_ylabel(r'$MSE/rad^{2}$', fontsize=24)
  ax.set_xlim([mse_all.SNR.min(), mse_all.SNR.max()])

  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True,ncols=3, fontsize=24)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.4)
  ax.grid(which='major', alpha=0.4)
  ax.tick_params(axis='both', which='major', labelsize=24)
  ax.tick_params(axis='both', which='minor', labelsize=24)
  plt.tight_layout()
  plt.show()