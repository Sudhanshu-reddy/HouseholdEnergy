import pickle
import pandas as pd

# Load saved model
with open("energy_model.pkl", "rb") as f:
    model = pickle.load(f)

# Test prediction
sample = pd.DataFrame([[0.1, 240, 10, 0, 0, 0, 12]], columns=[
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "hour"
])
pred = model.predict(sample)
print("Predicted Global Active Power:", pred[0])
