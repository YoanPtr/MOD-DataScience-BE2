import pandas as pd
from scipy.io import arff
import numpy as np
import os
# Fonction pour convertir un fichier ARFF en CSV
def convert_arff_to_csv(arff_file, csv_file):
    data, meta = arff.loadarff(arff_file)
    
    data = np.array([[str(val, 'utf-8') if isinstance(val, bytes) else val for val in row] for row in data])

    df = pd.DataFrame(data, columns=[name for name in meta.names()])
    
    df.to_csv(csv_file, index=False)

if __name__=='__main__': 

    folder = '/home/yoan/Documents/Centrale/4A/MOD/Intro Data Science/BE1/Rendu BE1/data/'

    for file in os.listdir(folder):
        if file.endswith('.arff'):
            arff_file = os.path.join(folder, file)
            csv_file = arff_file[:-5] + '.csv' 
            convert_arff_to_csv(arff_file, csv_file)
            os.remove(arff_file)
