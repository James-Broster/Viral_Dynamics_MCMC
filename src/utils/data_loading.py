import pandas as pd
import numpy as np

def load_data():
    data_fatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0],
        'virusload': [32359.37, 15135612, 67608298, 229086765, 245470892, 398107171, 213796209, 186208714, 23988329, 630957.3, 4265795, 323593.7, 53703.18, 141253.8]
    })
    data_nonfatal = pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'virusload': [165958.7, 52480.75, 2754229.0, 3548134.0, 1288250.0, 1584893.0, 199526.2, 371535.2, 107151.9, 14791.08, 31622.78, 70794.58, 7413.102, 9772.372, 5623.413]
    })
    time_fatal = data_fatal['time'].values
    time_nonfatal = data_nonfatal['time'].values
    observed_data_fatal = np.log10(data_fatal['virusload'].values).reshape(-1, 1)
    observed_data_nonfatal = np.log10(data_nonfatal['virusload'].values).reshape(-1, 1)
    return time_fatal, observed_data_fatal, time_nonfatal, observed_data_nonfatal