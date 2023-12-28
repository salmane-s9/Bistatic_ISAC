"""
This script plots the performance of the MLP Classfier architecture
for predictiing the number of targets in one peak
Simulation data are obtained by running the ClassifyNumberTargets.m MATLAB script and saved as mat files.
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
from models import HH2ComplexMLPClassifier
from loaders import ISAC_rav_HH_Classification_Dataset, ToComplexMLPClassifierInputsTensor
from trainer import train_classifier, compare_classification_models

import matplotlib
matplotlib.rcParams['font.family'] = "DejaVu Sans"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

style_label = 'seaborn-v0_8-deep'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


# Creating the Dataset 
train_snr_values = [-10, -5, 5, 15]
for snr_val in train_snr_values:
    data_path = f'./data/Bistatic_data_with_ToA_Nrtargets_classif/Bistatic_data_with_ToA_Nrtargets_classif_{snr_val:d}_dB'
    csv_path = f'./csv_data/Bistatic_data_with_ToA_Nrtargets_classif_{snr_val:d}_dB_df.csv'

    if ((not os.path.exists(csv_path)) and (os.path.exists(data_path))):
        print(f'SNR: {snr_val}\n')
        df_sim = create_dataset(data_path=data_path,  save_dir=csv_path,
                                columns_to_include=['TimeChannelEstimate'],
                                include_2D_3D=True)

# Creating Training DataFrame
all_sim_df = pd.DataFrame()
snr_bar = tqdm.tqdm(train_snr_values)
snr_bar.set_description("Creating Training DataFrame")
for snr_value in snr_bar:
   path_csv = f'./csv_data/Bistatic_data_with_ToA_Nrtargets_classif_{snr_value:d}_dB_df.csv'
   if os.path.exists(path_csv):
      tmp_df = pd.read_pickle(path_csv)
      tmp_df['TimeChannelEstimate_rav'] = tmp_df.TimeChannelEstimate.apply(np.ravel)
      all_sim_df = pd.concat((all_sim_df,tmp_df), ignore_index=True)

# Creating Training DataLoaders 
q_values = all_sim_df.q.explode().values
idx_train, idx_test = train_test_split(range(all_sim_df.shape[0]),test_size=0.02, random_state=2022, stratify=all_sim_df.q.explode())
all_sim_df['SET'] = 'TRAIN'
all_sim_df.loc[idx_test, ['SET']] = 'TEST'

df_train = all_sim_df[all_sim_df.SET == 'TRAIN'].copy(deep=True)
df_test = all_sim_df[all_sim_df.SET == 'TEST'].copy(deep=True)

LABEL = 'q'
N_max_targets = all_sim_df.q.explode().max()
isac_train_dataset = ISAC_rav_HH_Classification_Dataset(dataframe=df_train,
                                    input_column='TimeChannelEstimate_rav',
                                    label=LABEL, 
                                    transform=transforms.Compose([ToComplexMLPClassifierInputsTensor()]),
                                    )
isac_train_dataloader = DataLoader(isac_train_dataset, batch_size=128, shuffle=True, num_workers=0)

# Test dataset
isac_test_dataset = ISAC_rav_HH_Classification_Dataset(dataframe=df_test,
                                    input_column='TimeChannelEstimate_rav',
                                    label=LABEL,
                                    transform=transforms.Compose([ToComplexMLPClassifierInputsTensor()]),
                                    )
isac_test_dataloader = DataLoader(isac_test_dataset, batch_size=128, shuffle=True, num_workers=0)


model_name = f'Bistatic_3D_NEW_Nr_Targets_classifier_mixeddB'
PATH = f'./models/{model_name}.pt'
if not os.path.exists(PATH):
    N_OUTPUTS = N_max_targets
    N_EPOCHS = 10
    LR = 8e-5
    complex_model = HH2ComplexMLPClassifier(input_size=80, device=device, n_outputs=N_OUTPUTS)
    # Train the model
    complex_model = complex_model.to(device)
    net, history_test = train_classifier(
                            complex_model ,
                            isac_train_dataloader,
                            isac_test_dataloader,
                            epochs=N_EPOCHS,
                            lr=LR,
                            plot_save_dir=f'./results/{model_name}.png',
                            save_directory=None,
                            n_epochs_save_model=280,
                            use_schedulers=True,
                            milestones=[7],
                        )

    # Save the model
    print('Saving the model')
    torch.save(net.state_dict(), PATH)


# Creating Test set 
path_classif_test = './data/Bistatic_data_with_ToA_Nrtargets_classif_test'
csv_classif_test = './csv_data/Bistatic_data_with_ToA_Nrtargets_classif_test_df.csv'
if ((not os.path.exists(csv_classif_test)) and (os.path.exists(path_classif_test))):
    df_sim = create_dataset(data_path=path_classif_test,
                            save_dir=csv_classif_test,
                            columns_to_include=['TimeChannelEstimate'],
                            include_2D_3D=True)

all_snr_df = pd.read_pickle(csv_classif_test)
all_snr_df['TimeChannelEstimate_rav'] = all_snr_df.TimeChannelEstimate.apply(np.ravel)
all_snr_df = all_snr_df.explode(['snr'])

# Predicting on the Test set
N_OUTPUTS = all_snr_df.q.explode().max()
PATH = f'./models/Bistatic_3D_NEW_Nr_Targets_classifier_mixeddB.pt'
LABEL = 'q'
dict_models = {}
tmp_alldb_model = HH2ComplexMLPClassifier(input_size=80, device=device, n_outputs=N_OUTPUTS)
tmp_alldb_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
tmp_alldb_model.eval()

dict_models[f'MLP_Classifier_alldB'] = (tmp_alldb_model, partial(ISAC_rav_HH_Classification_Dataset, input_column='TimeChannelEstimate_rav',
                                                                 label=LABEL, transform=transforms.Compose([ToComplexMLPClassifierInputsTensor()])), 
                                        'single-input', 'multi_label_target')
df_accuracy_mean = compare_classification_models(all_snr_df, list_models=dict_models, label=LABEL, return_df=True, 
                                                 aggregate_function=np.mean, plot_per_model=True, legend_half=True,
                                                 plot=False)
accuracy_all = df_accuracy_mean.copy()
accuracy_all = accuracy_all[accuracy_all.SNR.isin(np.arange(-5, 21, step=2))]
with plt.style.context(style_label, {'grid.linestyle': '---'}):

  fix, ax = plt.subplots(figsize=(14, 8), layout='constrained')
  sns.lineplot(data=accuracy_all, x="SNR", y="Accuracy", hue="Model", ax=ax, style="Model", markersize=16, markers=True, dashes=True, errorbar=None, estimator=np.mean, linewidth=3)
  ax.set_xlabel(r'$SNR [dB]$', fontsize=24)
  ax.set_ylabel(r'$Accuracy$', fontsize=24)
  ax.set_xlim([accuracy_all.SNR.min(), accuracy_all.SNR.max()])
  ax.set_ylim([accuracy_all.Accuracy.min()-5, accuracy_all.Accuracy.max()+5])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), fancybox=True, shadow=True,ncols=np.ceil(accuracy_all.Model.nunique()/2), fontsize=24)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.8)
  ax.grid(which='major', alpha=0.8)
  ax.tick_params(axis='both', which='major', labelsize=24)
  ax.tick_params(axis='both', which='minor', labelsize=24)
  plt.tight_layout()
  plt.show()