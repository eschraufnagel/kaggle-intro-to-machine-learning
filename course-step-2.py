import pandas as pd
from datetime import datetime

melbourne_file_path = 'input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
print(melbourne_data["Car"].mean().round())
print(datetime.now().year - melbourne_data["YearBuilt"].max())