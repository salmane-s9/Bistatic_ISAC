# Bistatic_ISAC #
 This is the code base for our paper "Complex Neural Network based Joint AoA and AoD Estimation for Bistatic ISAC". 
 
 Authors: Salmane Naoumi, Ahmad Bazzi, Roberto Bomfin, Marwa Chafii

- - - -

## Requirements ##

Requirements are listed in the file **requirements.txt**
```bash
pip install -r requirements.txt
```

## Simulation Data ##

Simulation data are generated using MATLAB and saved as .mat files  
2D Estimator and CRB results are also obtained using the MATLAB scripts **CRB3D.m** and **Estim2D_new**  
The scripts to run in MATLAB to generate the simulation data for training are:  
- SNRAnalysis.m  
- ClassifyNumberTargets.m  
- Compare1TargetPeak.m  
- Compare2TargetsPeak.m  

## Joint Estimation Results ##

![alt text](https://github.com/salmane-s9/Bistatic_ISAC/blob/main/images/image_model.png?raw=true =600*500)
<!-- <img src="drawing.jpg" alt="drawing" width="200"/> -->

### SNR Analysis ###  

This script plots the performance of the MLP architecture trained at different SNR values and used for prediction on a range of Test SNRs (Figure. 4)
```bash
python snr_analysis.py 
```

### Number of Targets classification ###  

This script plots the performance of the MLP Classfier architecture for predictiing the number of targets in one peak (Figure. 5)
```bash
python classif_ntargets.py 
```

### Joint AoA-AoD predictions ###  

This script plots the performance comparison of MLP and convolutional networks with the parametric 2D Estimation algorithm for AoA estimation (Figure. 6)
```bash
python compare_1target_peak.py 
```
This script plots the performance comparison of MLP with the 2D Estimation algorithm for both AoA and AoD angles for settings with two targets per peak (Figure. 7)
```bash
python compare_2targets_peak.py 
```
