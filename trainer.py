import copy
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import torch
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

style_label = 'seaborn-v0_8-deep'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_SNR_analysis(cnnnetwork, data, test_data, epochs=100, test_metric='rad_mse', lr=1e-5, multi_target=False, multi_label=False, plot_save_dir=None, save_directory=None, 
          style_label='seaborn-v0_8-deep', n_epochs_save_model=100, use_schedulers=False, n_outputs=None):
  
  opt = torch.optim.Adam(cnnnetwork.parameters(),lr=lr)
  criterion = nn.MSELoss()
  if use_schedulers:
    scheduler1 = ExponentialLR(opt, gamma=0.9)
    scheduler2 = MultiStepLR(opt, milestones=[20, 50, 100], gamma=0.5)

  history_ae = []
  for epoch in tqdm.tqdm(range(epochs)):
      cnnnetwork.train()
      for sample in data:

          x = sample['input'].to(device)
          y = sample['target'].to(device)

          opt.zero_grad()
          y_hat = cnnnetwork(x)
          loss = ((y - y_hat)**2).sum()
          loss.backward()
          opt.step()

      
      if use_schedulers:
        scheduler1.step()
        scheduler2.step()
      
      cnnnetwork.eval()

      Y_test = []
      Y_pred = []
      for test_sample in test_data:
        
        x_test = test_sample['input'].to(device)
        y_test = test_sample['target']
        y_pred = cnnnetwork(x_test)
        y_pred = y_pred.to('cpu').detach()

        Y_test.append(y_test)
        Y_pred.append(y_pred)

      y_test = torch.concatenate(Y_test)
      y_pred = torch.concatenate(Y_pred)
      
      if test_metric == 'rad_mse':

        if multi_label and multi_target:
            mse_results = ((y_test - y_pred)**2)/((180/np.pi)**2)
            aoa_results = mse_results[:,:int(n_outputs/2)].mean(axis=0)
            aod_results = mse_results[:,int(n_outputs/2):].mean(axis=0)
            mse = (aoa_results, aod_results)
        else:
            if multi_label or multi_target:
                mse = ((y_test - y_pred)**2).mean(axis=0)/((180/np.pi)**2) 
            else:
                mse = ((y_pred - y_test)**2).sum()  
      else:
        raise ValueError('test metric not supported')
          
      if multi_label and multi_target:
         mse = [_mse.detach().numpy() for _mse in mse]
         history_ae.append(([mse[0][i] for i in range(int(n_outputs/2))], [mse[1][i] for i in range(int(n_outputs/2))]))
      else:
        if multi_label:
          mse = mse.detach().numpy()
          history_ae.append((mse[0], mse[1]))
        elif multi_target:
          mse = mse.detach().numpy()
          history_ae.append([mse[i] for i in range(n_outputs)])
        else:
          mse = float(mse)
          history_ae.append(mse)
        
      if (save_directory is not None) and (epoch >= n_epochs_save_model):
        torch.save(cnnnetwork.state_dict(), f'{save_directory}_epoch_{epoch}.pt')

  with plt.style.context(style_label):
    x = np.arange(1, epochs+1)
    plt.figure(figsize=(12, 8), layout='constrained')

    if multi_label:
      if multi_target:
        for i in range(int(n_outputs/2)):
          plt.semilogy(x, np.array([hist_mse[0][i] for hist_mse in history_ae]), label=f'Test set MSE (AoA) (Target {i})')    
          plt.semilogy(x, np.array([hist_mse[1][i] for hist_mse in history_ae]), label=f'Test set MSE (AoD) (Target {i})')    

      else:
        plt.semilogy(x, np.array([hist_mse[0] for hist_mse in history_ae]), label='Test set MSE (AoA)')
        plt.semilogy(x, np.array([hist_mse[1] for hist_mse in history_ae]), label='Test set MSE (AoD)')
    elif multi_target and (not multi_label):
      for i in range(n_outputs):
        plt.semilogy(x, np.array([hist_mse[i] for hist_mse in history_ae]), label=f'Test set MSE (Target {i})')    
    else:
      plt.semilogy(x, np.array(history_ae), label='Test set MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.xlim([1, epochs])
    plt.title("Test Set MSE")
    if plot_save_dir is not None:
      plt.savefig(plot_save_dir)
    plt.legend()
    plt.grid(True)
    plt.show()
  
  return cnnnetwork, history_ae


def snr_analysis_comparison(dataframe_snrs, list_models, label, device=device, style_label=style_label, include_2D=False, include_3D=False,
                            return_df=False, show_ci=True, normalized_mse=False, multi_target=False, aggregate_function=np.mean, title=None,
                            legend_half=True, plot=True):

  list_models_2 = copy.deepcopy(list_models)
  mse_values = list()
  if include_2D:
    list_models_2['2D Estimation'] = None
  if include_3D:
    list_models_2['3D Estimation'] = None
  models_bar = tqdm.tqdm(list_models_2, position=0, leave=False)
  unique_snr = list(sorted(dataframe_snrs.snr.unique()))
  for model_name in models_bar:
    if model_name=="2D Estimation":
      snrs_bar = tqdm.tqdm(unique_snr, position=0, leave=True)
      for snr in snrs_bar:
        snrs_bar.refresh()
        snrs_bar.set_description(f"{model_name} ====> SNR = {snr} dB")
        tmp3d = dataframe_snrs[dataframe_snrs.snr == snr][[label, f'{label}_est_2D']]
        if multi_target:
            y_test = np.sort(np.vstack(tmp3d[label].values))
            y_pred = np.sort(np.vstack(tmp3d[f'{label}_est_2D'].values))

            if normalized_mse:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
            else:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)

            mse_results = np.mean(mse_results, axis=1)
        else:
          y_test = tmp3d[label].explode().values
          y_pred = tmp3d[f'{label}_est_2D'].explode().values
          if normalized_mse:
            mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
          else:
            mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
        tmp_mse = pd.DataFrame(data={'MSE': mse_results.ravel().tolist()})
        tmp_mse['SNR'] = snr
        tmp_mse['Model'] = model_name
        mse_values.append(tmp_mse)

    elif model_name=="3D Estimation":
      snrs_bar = tqdm.tqdm(unique_snr, position=0, leave=True)
      for snr in snrs_bar:
        snrs_bar.refresh()
        snrs_bar.set_description(f"{model_name} ====> SNR = {snr} dB")
        tmp3d = dataframe_snrs[dataframe_snrs.snr == snr][[label, f'{label}_est_3D']]
        if multi_target:
            y_test = np.sort(np.vstack(tmp3d[label].values))
            y_pred = np.sort(np.vstack(tmp3d[f'{label}_est_3D'].values))
            if normalized_mse:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
            else:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
            mse_results = np.mean(mse_results, axis=1)
        else:
          y_test = tmp3d[label].explode().values
          y_pred = tmp3d[f'{label}_est_3D'].explode().values
          if normalized_mse:
            mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
          else:
            mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)

        tmp_mse = pd.DataFrame(data={'MSE': mse_results.ravel().tolist()})
        tmp_mse['SNR'] = snr
        tmp_mse['Model'] = model_name
        mse_values.append(tmp_mse)

    else:
      snrs_bar = tqdm.tqdm(unique_snr, position=0, leave=True)
      model, dataset_loader, nature_input, nature_label = list_models[model_name]
      model = model.to(device)
      model.eval()

      for snr in snrs_bar:
        snrs_bar.refresh()
        snrs_bar.set_description(f"{model_name} ====> SNR = {snr} dB")
        dataset = dataset_loader(dataframe=dataframe_snrs[dataframe_snrs.snr == snr].reset_index(drop=True))
        dataset = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

        Y_pred = []
        Y_test = []

        # test_sample = next(iter(dataset))
        for test_sample in dataset:
          if nature_input=='multi-input':
            x_test = test_sample['input']
            x_test = [x_.to(device) for x_ in x_test]
          elif nature_input=='single-input':
            x_test = test_sample['input'].to(device)

          y_test = test_sample['target']
          y_pred = model(x_test)
          y_pred = y_pred.to('cpu').detach()
          y_test = y_test.numpy()
          y_pred = y_pred.numpy()

          Y_pred.append(y_pred)
          Y_test.append(y_test)

        y_test = np.vstack(Y_test)
        y_pred = np.vstack(Y_pred)
        y_pred = np.round(y_pred, 2).astype(float)

        if nature_label=='multi_label_target':
          n_outputs = int(y_test.shape[1]/2)
          if label=='AoA':
            mse_results = ((np.sort(y_test[:,:n_outputs]) - np.sort(y_pred[:,:n_outputs]))**2) / ((180/np.pi)**2)
          else:
            mse_results = ((np.sort(y_test[n_outputs:]) - np.sort(y_pred[n_outputs:]))**2) / ((180/np.pi)**2)

          mse_results = np.mean(mse_results, axis=1)

        else:
          if multi_target:
            if normalized_mse:
              mse_results = ((np.sort(y_test) - np.sort(y_pred))**2) / ((180/np.pi)**2)
            else:
              mse_results = ((np.sort(y_test) - np.sort(y_pred))**2) / ((180/np.pi)**2)
            mse_results = np.mean(mse_results, axis=1)
          else:
            if nature_label=='multi_label':
              y_test = y_test[:,0 if label=='AoA' else 1]
              y_pred = y_pred[:,0 if label=='AoA' else 1]

            if normalized_mse:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
            else:
              mse_results = ((y_test - y_pred)**2) / ((180/np.pi)**2)
        tmp_mse = pd.DataFrame(data={'MSE': mse_results.ravel().tolist()})
        tmp_mse['SNR'] = snr
        tmp_mse['Model'] = model_name
        mse_values.append(tmp_mse)

  print('\n\n')
  mse_all = pd.concat(mse_values, ignore_index=True)

  if plot:
    with plt.style.context(style_label, {'grid.linestyle': '---'}):
      # , {'grid.linestyle': '--'}
      fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
      sns.lineplot(data=mse_all, x="SNR", y="MSE", hue="Model", ax=ax, style='Model', markersize=10, markers=True, dashes=True, errorbar='ci' if show_ci else None, estimator=aggregate_function)
      ax.set_yscale('log')
      ax.set_xlabel(r'$SNR [dB]$', fontsize=12)
      ax.set_ylabel(r'$MSE/rad^{2}$', fontsize=12)
      ax.set_xlim([mse_all.SNR.min(), mse_all.SNR.max()])

      if legend_half:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), fancybox=True, shadow=True,ncols=np.ceil(mse_all.Model.nunique()/2), fontsize=12)
      else:
        ax.legend(bbox_to_anchor=(1.1, 1.02), fancybox=True, shadow=True,ncols=1, fontsize=12)

      # Or if you want different settings for the grids:
      ax.grid(which='both')
      ax.grid(which='minor', alpha=0.1)
      ax.grid(which='major', alpha=0.7)
      ax.tick_params(axis='both', which='major', labelsize=11)
      ax.tick_params(axis='both', which='minor', labelsize=11)

      if title is not None:
        ax.set_title(title, fontsize=14, fontfamily='DejaVu Sans', fontweight='normal', style="italic")
      plt.tight_layout()
      plt.show()

  if return_df:
    return mse_all