##
# Extract RL training tuples
# (S_t, A_t, S_{t+1}, R_{t+1}, Done)
##
# Consolidated from extract_rl_tuples.ipynb
##

import os
import numpy as np
from datetime import datetime, date, time
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle

DATA_DIR = '../data'
INPUT_FILE = 'MIMICtable.csv'
OUTPUT_FILE = 'rl_samples.pkl'
infile = os.path.join(DATA_DIR, INPUT_FILE)
outfile = os.path.join(DATA_DIR, OUTPUT_FILE)

cols = 'bloc,icustayid,charttime,gender,age,elixhauser,re_admission,died_in_hosp,died_within_48h_of_out_time,mortality_90d,delay_end_of_record_and_discharge_or_death,Weight_kg,GCS,HR,SysBP,MeanBP,DiaBP,RR,SpO2,Temp_C,FiO2_1,Potassium,Sodium,Chloride,Glucose,BUN,Creatinine,Magnesium,Calcium,Ionised_Ca,CO2_mEqL,SGOT,SGPT,Total_bili,Albumin,Hb,WBC_count,Platelets_count,PTT,PT,INR,Arterial_pH,paO2,paCO2,Arterial_BE,Arterial_lactate,HCO3,mechvent,Shock_Index,PaO2_FiO2,median_dose_vaso,max_dose_vaso,input_total,input_4hourly,output_total,output_4hourly,cumulated_balance,SOFA,SIRS'.split(
    ',')
types = dict((n, np.float64) for n in cols)
mimic3 = pd.read_csv(infile, header=0, usecols=cols,
                     dtype=types)  # type(mimic3)
# prepare data remove patients who died in ICU during data collection period # Nature paper use '&' and remove 850 only
id_ICU_died = mimic3.query(
    'died_within_48h_of_out_time == 1 or delay_end_of_record_and_discharge_or_death < 24').icustayid.unique()

rawdata = mimic3[~mimic3.icustayid.isin(id_ICU_died)]

# 47 columns of state variables
bin_cols = ['gender', 'mechvent', 'max_dose_vaso', 're_admission']
norm_cols = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb',
             'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance']
log_cols = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili',
            'INR', 'input_total', 'input_4hourly', 'output_total', 'output_4hourly']

icustayid_list = mimic3['icustayid']

# state variables 47
# raw values
states_raw = mimic3[bin_cols + norm_cols + log_cols]
states_raw.head()

# normalized for clustering in L2 distance
states_norm = pd.DataFrame(
    data=None, columns=states_raw.columns, index=states_raw.index)
# in python for loop is faster than pandas.DataFrame.apply()
for col in bin_cols:
    states_norm[col] = states_raw[col] - 0.5
for col in norm_cols:
    df = states_raw[col]
    states_norm[col] = (df - df.mean())/df.std()
for col in log_cols:
    df = np.log(0.1+states_raw[col])
    states_norm[col] = (df - df.mean())/df.std()

# max dose norad
states_norm['max_dose_vaso'] = np.log(states_norm['max_dose_vaso']+.6)
# increase the weight of variable 'input_4hourly'
# (bin_cols + norm_cols + log_cols)[44]
states_norm['input_4hourly'] = 2 * states_norm['input_4hourly']

# hyper-parameters
n_models = 3       # 500
test_frac = 0.2
km_frac = 0.25
#
n_states = 750   # number of states if discretized
n_act = [5, 5]  # number of actions in each dimension if discretized
n_actions = np.prod(n_act)  # number of actions if discretized

#########################################
##
# discreteize continuous action space
##

# Two actions: IV fluid: 'input_4hourly' and Vasopressor: 'max_dose_vaso'
# mimic3['input_4hourly'].describe()
# mimic3['max_dose_vaso'].describe()

iv_bins = [-np.inf, 0, 50, 180, 530, np.inf]
vaso_bins = [-np.inf, 0, 0.08, 0.22, 0.45, np.inf]

mimic3['act_iv'] = pd.cut(mimic3['input_4hourly'],
                          bins=iv_bins, labels=range(5))
mimic3['act_vaso'] = pd.cut(mimic3['max_dose_vaso'],
                            bins=vaso_bins, labels=range(5))

# generate action code = (act_vi * 5 + act_vaso)
# mimic3['action'] = list(zip(mimic3.act_vi, mimic3.act_vaso))
mimic3['action'] = mimic3['act_iv'].cat.codes * \
    5 + mimic3['act_vaso'].cat.codes
mimic3[['act_iv', 'act_vaso', 'action']].head()

train_ids, test_ids = train_test_split(
    mimic3.icustayid.unique(), test_size=test_frac)
states_norm_train = states_norm[mimic3.icustayid.isin(train_ids)]

# load pickle file
infile_kmeans = os.path.join(DATA_DIR, 'states_kmeans.pkl')
states_kmeans = pickle.load(open(infile_kmeans, 'rb'))

mimic3['step'] = 1
mimic3['state'] = states_kmeans.predict(states_norm)
mimic3['next_state'] = 750  # default assign successful discharge
mimic3['trajectory'] = 0  # default trajectory
mimic3['done'] = False

# may not be necessary if produced in order
mimic3.sort_values(by=['icustayid', 'bloc'], inplace=True)

'''
Reward engineering
Info. Sources:
[1] https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/in-depth/blood-pressure/art-20050982
[2] https://medlineplus.gov/ency/article/003855.htm
[3] Arterial base excess: https://en.wikipedia.org/wiki/Base_excess
[4] PaO2_FiO2: https://litfl.com/pao2-fio2-ratio/
[5] INR: https://www.mayoclinic.org/tests-procedures/prothrombin-time/about/pac-20384661
[6] Blood clotting test: PTT, PT, INR: https://emedicine.medscape.com/article/2086058-overview#a1
[7] Hemoglobin test: https://www.mayoclinic.org/tests-procedures/hemoglobin-test/about/pac-20385075
[8] Sepsis: https://www.nursingcenter.com/ncblog/march-2017/laboratory-signs-of-sepsis
'''

# ['dangerous_low', 'normal_low', 'normal_high', 'dangerous_high']
# currently I only use ['normal_low', 'normal_high'] as the normal range, out of which will be penalized
# with some relaxations
# need help from critical care givers

VITAL_RANGES = {
    'HR': [np.nan, 60, 100, 130],
    'SysBP': [np.nan, 90, 140, 180],  # Systolic blood pressure
    'MeanBP': [np.nan, 70, 100, np.nan],
    'DiaBP': [np.nan, 60, 90, 120],  # Diastolic blood pressure
    'RR': [np.nan, 12, 20, 30],  # Respiratory Rate
    # O2 saturation pulseoxymetry, Blood oxygen levels below 80 percent may compromise organ function, such as the brain and heart, and should be promptly addressed.
    'SpO2': [90, 95, 100, np.nan],
    'Temp_C': [np.nan, 36.1, 38, np.nan],
    'FiO2_1': [],  # fraction of inspired oxygen
    'Potassium': [np.nan, 3.6, 5.2, 6],  # mmol/L
    'Sodium': [np.nan, 135, 145, np.nan],  # mEq/L
    'Chloride': [np.nan, 97, 107, np.nan],  # mEq/L
    'Glucose': [np.nan, 100, 125, np.nan],  # mg/dL
    'BUN': [np.nan, 7, 20, np.nan],  # Blood urea nitrogen
    'Creatinine': [np.nan, 0.55, 1.15, np.nan],
    'Magnesium': [np.nan, 1.5, 2.5, np.nan],
    'Calcium': [np.nan, 8.5, 10.5, np.nan],
    # In adults, a level of 4.64 to 5.28 milligrams per deciliter (mg/dL) is normal, but this range does not match the data
    'Ionised_Ca': [],
    'CO2_mEqL': [np.nan, 23, 29, np.nan],
    'SGOT': [np.nan, 5, 40, np.nan],
    'SGPT': [np.nan, 7, 56, np.nan],
    'Total_bili': [np.nan, 1.71, 20.5, np.nan],  # Total Bilirubin
    'Albumin': [np.nan, 3, 6, np.nan],
    # Hemoglobin, For men, 13.5 to 17.5 grams per deciliter. For women, 12.0 to 15.5 grams per deciliter.
    'Hb': [np.nan, 12, 1.5, np.nan],
    'WBC_count': [np.nan, 4.5, 11, np.nan],
    'Platelets_count': [np.nan, 150, 450, np.nan],
    # Partial thromboplastin time. The reference range of the PTT is 60-70 seconds.
    'PTT': [np.nan, 40, 80, np.nan],
    'PT': [np.nan, 9, 15, np.nan],  # prothrombin time
    # International normalized ratio. If you are not taking blood thinning medicines, such as warfarin, the normal range for your PT results is: 11 to 13.5 seconds. INR of 0.8 to 1.1.
    'INR': [np.nan, 0.5, 1.5, np.nan],
    'Arterial_pH': [7.3, 7.35, 7.45, np.nan],
    'paO2': [np.nan, 75, 100, np.nan],  # Partial pressure of oxygen
    'paCO2': [np.nan, 38, 42, np.nan],  # Partial pressure of carbon dioxide
    'Arterial_BE': [np.nan, -2, 2, np.nan],  # Arterial base excess
    'Arterial_lactate': [np.nan, 0.5, 2.5, 4],
    'HCO3': [np.nan, 22, 28, np.nan],  # mEq/L Bicarbonate
    # 'PaO2_FiO2': [100, 300, 2900, np.nan] # max value in the dataset is 2890.4761904761895. # PF ratio <333 mmHg if age <50y or PF ratio <250mmHg if age >50y; the Berlin definition of ARDS (PF ratio <300mmHg), and correlates with mortality: mortality 27% if (200, 300], 32% if (100, 200], 45% if (0, 100].  # OverflowError: math range error
    # ---------------------------
    # 'SaO2':[np.nan, 0.94, 1, np.nan]
}

# IMPORTANT_VITALS = 'Arterial_lactate, Creatinine, Total_bili, Glucose, WBC_count, Platelets_count, PTT, PT, INR'.split(',') # PaO2_FiO2,
IMPORTANT_VITALS = 'HR,SysBP,MeanBP,DiaBP,RR,SpO2,Arterial_lactate,Creatinine,Total_bili,Glucose,WBC_count,Platelets_count,PTT,PT,INR'.split(
    ',')  # PaO2_FiO2,


# First time only use a small number of criterions.
VITAL_IMPORTANCE = dict.fromkeys(VITAL_RANGES.keys(), 0)

for vitals in IMPORTANT_VITALS:
    VITAL_IMPORTANCE[vitals] = 1

# generate 'trajectory' and 'next_state' and ...
# 0 - 749 cluster
# ... add two absorbing states: 751 (die in hospital), 750 (discharge)

DECEASED = 751
DECEASED_REWARD = -50
DISCHARGED = 750
SURVIVAL_REWARD = 50

#VITS = 'HR SysBP MeanBP DiaBP RR SpO2'.split()


def getVitalReward(row, regulateVitals, vital_weights):
    total_reward = 0
    # for vit in VITS:
    for vit in regulateVitals:
        if len(regulateVitals[vit]) > 2:
            lower = regulateVitals[vit][1]
            upper = regulateVitals[vit][2]
            mid = (upper - lower)/2.0
            importance = vital_weights[vit]  # regulateVitals[vit][4]
            diff2 = float(upper-lower)/2
            scale = 1/((math.exp(diff2)-1)/(1+math.exp(diff2)))
            val = row[vit]
            if (val > upper) or (val < (lower)):
                reward = -0.125*importance*scale * \
                    (1/(1+math.exp(lower-val)) - 1 /
                     (1+math.exp(upper-val)) - (1/scale))
                total_reward -= reward
    return total_reward


trajectory = 1  # one trajectory for one patient
step = 1  # the number of step in each trajectory

print('Processing patient trajectory #' + str(trajectory))

for i, row in mimic3.iterrows():

    mimic3.loc[i, 'trajectory'] = trajectory

    if i+1 == mimic3.shape[0]:
        mimic3.loc[i, 'step'] = step  # no next df line
        mimic3.loc[i, 'next_state'] = DECEASED if row['died_in_hosp'] == 1 else DISCHARGED
        mimic3.loc[i, 'done'] = True
        mimic3.loc[i, 'reward'] = DECEASED_REWARD if row['mortality_90d'] == 1 else SURVIVAL_REWARD
        print('End of dataframe at row ' + str(i))
    elif mimic3.loc[(i+1), 'bloc'] > mimic3.loc[i, 'bloc']:
        mimic3.loc[i, 'step'] = step
        step += 1  # stay in the same trajectory, step ++
        mimic3.loc[i, 'next_state'] = mimic3.loc[i+1, 'state']
        mimic3.loc[i, 'reward'] = getVitalReward(
            mimic3.loc[i+1, :], VITAL_RANGES, VITAL_IMPORTANCE)
    elif mimic3.iloc[i+1]['bloc'] == 1:
        mimic3.loc[i, 'step'] = step
        step = 1  # prepare for next trajectory
        mimic3.loc[i, 'next_state'] = DECEASED if row['died_in_hosp'] == 1 else DISCHARGED
        mimic3.loc[i, 'done'] = True
        mimic3.loc[i, 'reward'] = DECEASED_REWARD if row['mortality_90d'] == 1 else SURVIVAL_REWARD
        trajectory = trajectory + 1
        print('Processing trajectory #' + str(trajectory))
    else:
        print('Error at entry' + str(i))

with open(outfile, 'wb') as file:
    pickle.dump(mimic3, file)
