import os
import tqdm
import scipy.io as sio
import pandas as pd
import numpy as np

def create_snr_analysis_dataset(data_path, save_dir, columns_to_include=None, include_2D_3D=False, include_CRB=False):

  def get_by_keyword(mat_struct, key_words):
    return dict((kw, mat_struct[kw][0,0].ravel()) for kw in key_words)

  files_dir = [file for file in os.listdir(data_path) if '(1)' not in file]

  df_columns = ['q', 'mc', 'snr', 'Nr', 'Nt', 'ChannelEstimate']
  if columns_to_include is not None:
    df_columns += columns_to_include
  
  df_columns += ['AoA', 'AoD', 'ToA_BS_Target', 'ToA_Target_radar', 'doppler_norm', 'distance_BS_Target', 'distance_Target_radar']
  if include_2D_3D:
    df_columns += ['AoA_est_2D', 'AoD_est_2D', 'AoA_est_3D', 'AoD_est_3D', 'ToA_est_3D']
  if include_CRB:
    # Extracting only bounds for AoA and AoD
    df_columns += ['phi', 'theta']
    
  df_sim = pd.DataFrame(columns=df_columns)
  
  for sim_file in tqdm.tqdm(files_dir):
      sim_dict = dict()
      simulation_data = sio.loadmat(os.path.join(data_path,sim_file), struct_as_record=True)
      simulation_data = simulation_data['GENDATA'][0][0]
      
      # simulation_data
      # General Parameters
      GeneralParameters = get_by_keyword(simulation_data['GeneralParameters'], ['q', 'mc', 'snr', 'Nr', 'Nt'])
      sim_dict.update(GeneralParameters)

      try:
        ChannelEstimate = simulation_data['ChannelEstimate']

        sim_dict.update(
            {
                'ChannelEstimate': ChannelEstimate.astype(np.cdouble),
            }
        )
      except:
        pass
  
      if columns_to_include is not None:

        sim_dict.update(
            dict((col, simulation_data[col].astype(np.cdouble)) for col in columns_to_include)
        )  

      # Targets Parameters 
      TargetsParameters = get_by_keyword(simulation_data['TargetsParameters'], ['AoA', 'AoD', 'ToA_BS_Target', 'ToA_Target_radar', 'doppler_norm',
                                                                                'distance_BS_Target', 'distance_Target_radar'])
      sim_dict.update(TargetsParameters)

      # 2D and 3D algorithms estimation
      if include_2D_3D:
        try:
          Estimated2DParameters = get_by_keyword(simulation_data['Estimated2DParameters'], ['AoA_est_2D', 'AoD_est_2D'])
          sim_dict.update(Estimated2DParameters)
          
          Estimated3DParameters = get_by_keyword(simulation_data['Estimated3DParameters'], ['AoA_est_3D', 'AoD_est_3D', 'ToA_est_3D'])
          sim_dict.update(Estimated3DParameters)
        except:
          pass
      
      if include_CRB:
        try:
          CRBResults = get_by_keyword(simulation_data['CRBResults'], ['phi', 'theta'])
          sim_dict.update(CRBResults)
          
        except:
          pass
              
      # Updating the DataFrame
      df_sim = df_sim.append(sim_dict, ignore_index=True)

  df_sim.to_pickle(save_dir)

  return df_sim 