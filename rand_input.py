import pandas as pd
import random
import os


def rand_output(path):

    random_output = pd.read_csv(os.path.join(path), sep="\t", engine='python')

    random_output['onset'] = [round(random.uniform(0, 10), 3)
                              for k in random_output.index]
    random_output['offset'] = [round(random.uniform(onset, 10), 3)
                               for onset in random_output['onset']]
    try:
            random_output.to_csv(r'export_dataframe.csv', index=None,
                                 header=True, sep="\t")
    except Exception:
            print("The file was not written to")
