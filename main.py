import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your new CSV file (make sure the file path is correct)
df = pd.read_csv('processed_data.csv')  # Replace 'your_file.csv' with your actual CSV file path

# Let's take a look at the first few rows of the data
print(df.head())

# Preprocess the data
# We'll use 'Decimal_Year' as the independent variable (X) and 'Anomaly' as the target variable (y)
X = df['Decimal_Year'].values.reshape(-1, 1)  # 'Decimal_Year' as the feature
y = df['Anomaly'].values  # 'Anomaly' as the target

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the training set results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='green', label='Training Data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Temperature Anomaly (Training Set)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.show()

# Visualizing the test set results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Temperature Anomaly (Test Set)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.show()

# Now, let's predict the temperature anomaly for a future year (e.g., 2050)
year = 2022
X_predict = np.array([year]).reshape(1, -1)  # Input the year you want to predict for
y_predict = regressor.predict(X_predict)

# Output the predicted anomaly for the year
print(f"The predicted temperature anomaly for the year {year} is: {y_predict[0]:.4f}°C")

# To calculate the absolute temperature:
# Assuming the temperature anomaly is relative to a base period (e.g., 1880), you would add the anomaly to the base temperature.
# Let's assume the base temperature (1880) is 14°C (this is just an example, you should adjust based on actual historical data).
base_temperature = 14.0  # Base temperature in 1880, you can adjust this as necessary
predicted_temperature_2050 = base_temperature + y_predict[0]
print(f"The predicted global temperature for the year  {year} is: {predicted_temperature_2050:.4f}°C")
