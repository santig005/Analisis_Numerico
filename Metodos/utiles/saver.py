import os
import numpy as np
from tabulate import tabulate

def dataframe_to_txt(dataframe, filename):
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    try:
        with open(f'resultados/{filename}.txt', 'w') as f:
            # Use tabulate to format the dataframe as a table
            table = tabulate(dataframe, headers='keys', tablefmt='pretty')
            f.write(table)
    except Exception as e:
        print("An error occurred while transcribing the dataframe:", str(e))