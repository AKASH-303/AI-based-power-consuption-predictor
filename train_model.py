# Simple training helper. Usage: python train_model.py data/your.csv
import sys
from model import train_on_csv
if len(sys.argv) < 2:
    print("Provide path to CSV: python train_model.py data/sample_power_data.csv")
else:
    mse = train_on_csv(sys.argv[1])
    print("Trained. MSE:", mse)
