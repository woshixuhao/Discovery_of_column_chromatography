# Discovery_of_column_chromatography
The data and code for the paper 'How Chemists Determine Column Chromatography Conditions by Thin-Layer Chromatography: A Rationale from Statistics and Machine learning'


# Environments
python==3.7.3  
rdkit==2020.09.1.0  
pandas==1.3.5  
numpy==1.21.6  
pytorch==1.11.0  
pubchempy==1.0.4  
mordred==1.2.0  
pysr==0.16.3  
scikit-learn==1.0.2  
scipy==1.7.3  
pybaobabdt==1.0.1

# The generated experimental dataset in this work 
* TLC_dataset.xlsx (The dataset of Rf values measured under different conditions and compounds, 4944 data)
* dataset_1013.xlsx (The dataset of retention volume during CC process in 4g column, 4998 data)
* dataset_25g_1218.xlsx (The dataset of retention volume during CC process in 25g column, 528 data)
* dataset_4+4g_1020.xlsx (The dataset of retention volume during CC process in 4g+4g column, 456 data)


# How to reproduce our work?
1. Install relevant packages (~0.5h)  
2. Run the Train_surrogate_model.py (~5 min for TLC prediction model and ~20 min for CC prediction model)
   This work utilize the GPU acceleration by Nvidia Gefore GTX TITAN  
3. Run the Discover_equations.py to discover the relationship between the TLC and CC (~5 min).
4. Run the transfer_interpret(25g).py and transfer_interpret(4g+4g).py for transfer learning (~10 min).


# Expected outputs
* surrogate models--model CC and model TLC in dir of "model_save". (These models can be used for prediction of Rf value and retention times)
* discovered equations--Rf_t1_mean.csv and Rf_t2_mean.csv, which are discovered using pysr to describe the relationship between TLC and CC.
