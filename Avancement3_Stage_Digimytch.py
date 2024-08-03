#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
# Load the power production dataset
file_path = r"C:\Users\najar\Downloads\PV_Elec_Gas3.csv"
data = pd.read_csv(file_path)


# In[2]:


# Load the weather dataset
weather_file_path = r"C:\Users\najar\Downloads\weather_in_Antwerp.csv\weather_in_Antwerp.csv"
weather_data = pd.read_csv(weather_file_path, delimiter=';')


# In[3]:


weather_data


# In[4]:


#creating a new feature that seems useful for plotting
data['Daily Production'] = 0  # or any other initial value

data['Daily Production'] = data['Cumulative_solar_power'].diff()


# In[5]:


# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Define the start and end dates
start_date = pd.to_datetime('01/01/2012', format='%d/%m/%Y')
end_date = pd.to_datetime('30/12/2019', format='%d/%m/%Y')

# Filtering the power data so it has the same range as weather_data
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Plotting the data
plt.figure(figsize=(14, 7))
plt.stem(data['date'], data['Daily Production'], use_line_collection=True)
plt.title('Daily Solar Power Production')
plt.xlabel('date')
plt.ylabel('Daily Production (kWh)')
plt.grid(True)

# Set the correct date format on the x-axis
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m/%Y'))
plt.gcf().autofmt_xdate()
plt.show()


# In[6]:


data


# In[7]:


duplicated_dataset = weather_data.copy()
# Create a 'Date' column by combining 'year', 'month', and 'day'
duplicated_dataset['Date'] = pd.to_datetime(duplicated_dataset[['year', 'month', 'day']])

# Drop the original 'year', 'month', and 'day' columns
duplicated_dataset= duplicated_dataset.drop(columns=['year', 'month', 'day'])


# In[8]:


duplicated_dataset.drop(columns={'Unnamed: 0'})


# In[9]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example function to calculate sunrise and sunset times
def sunrise_sunset(date):
    day_of_year = date.timetuple().tm_yday
    # Example calculation, replace with your accurate method
    sunrise_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=6, minutes=0) + timedelta(minutes=(30 * np.sin((day_of_year - 80) * 2 * np.pi / 365)))
    sunset_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=18, minutes=0) + timedelta(minutes=(-30 * np.sin((day_of_year - 80) * 2 * np.pi / 365)))
    return sunrise_time.time(), sunset_time.time()



duplicated_dataset = duplicated_dataset.copy()  # Create a copy to avoid SettingWithCopyWarning
duplicated_dataset.loc[:, 'Date'] = pd.to_datetime(duplicated_dataset['Date'], format='%Y-%m-%d')

# Generate date range for entire period 2012-01-01 to 2019-12-30, the same range as the other two datasets
date_range_full = pd.date_range(start='2012-01-01', end='2019-12-30')

# Create a DataFrame for sunrise and sunset times for each date in the range
set_rise = pd.DataFrame(date_range_full, columns=['Date'])

# Apply sunrise_sunset function to set_rise DataFrame to get Sunrise and Sunset times
set_rise['Sunrise'], set_rise['Sunset'] = zip(*set_rise['Date'].apply(sunrise_sunset))

# Merge duplicated_dataset with set_rise on 'Date'
merged_data = pd.merge(duplicated_dataset, set_rise, on='Date')

# Convert 'clock' column to datetime if it's not already
merged_data['clock'] = pd.to_datetime(merged_data['clock'])

# Extract the clock time from 'clock' column
merged_data['clock'] = merged_data['clock'].dt.time

# Filter rows to keep only those where 'clock' is between 'Sunrise' and 'Sunset' for each date
filtered_data1 = merged_data[
    (merged_data['clock'] >= merged_data['Sunrise']) &
    (merged_data['clock'] <= merged_data['Sunset'])
]

print(filtered_data1.head(100))
print(filtered_data1.columns)


# In[10]:


#remove the km/h,°C , mbar ....
def clean_numeric_column(column):
    if column.dtype == 'object':  # Check if the column is of object type (typically strings)
        return pd.to_numeric(column.str.replace(r'[^\d.]+', '', regex=True), errors='coerce')
    else:
        return pd.to_numeric(column, errors='coerce')  # Convert non-string columns directly

# Create a copy of filtered_data1 with only the columns to clean
filtered_data = filtered_data1[['temp', 'wind',  'humidity', 'barometer']].copy()

# Clean the relevant columns in filtered_data using clean_numeric_column function
filtered_data['temp'] = clean_numeric_column(filtered_data['temp'])
filtered_data['wind'] = clean_numeric_column(filtered_data['wind'])
filtered_data['humidity'] = clean_numeric_column(filtered_data['humidity'])
filtered_data['barometer'] = clean_numeric_column(filtered_data['barometer'])


# Include sunrise and sunset columns from filtered_data1 in filtered_data
filtered_data[['Date','clock','Sunrise', 'Sunset']] = filtered_data1[['Date','clock','Sunrise', 'Sunset']]
print(filtered_data)


# In[11]:


# Group by date and calculate daily means
daily_weather =filtered_data.groupby('Date').agg({
    'temp': 'mean',
    'wind': 'mean',
    'humidity': 'mean',
    'barometer': 'mean',
    'Sunrise': 'first',  # Keep the first occurrence of Sunrise time for each date
    'Sunset': 'first'
    
}).reset_index()


# In[12]:


daily_weather


# In[13]:


daily_weather2=daily_weather.copy()


# In[14]:


data


# In[15]:


# Merge the dataframes on the 'date' and 'Date' columns using an inner join
merged_data = pd.merge(daily_weather2, data[['date', 'Daily Production']], left_on='Date', right_on='date', how='inner')

# Assign the 'Daily Production' column from merged_data to daily_weather2
daily_weather2 = merged_data.drop(columns='date')

# Check the result
print(daily_weather2.head())


# In[16]:


daily_weather2


# In[17]:


daily_weather3=daily_weather2.copy()


# In[18]:


import pandas as pd
#here, we're trying to take the weather condition when the clock time is 13h20, or when its closest to it as it generally reflects the weather condition for the whole day
# Ensure 'Date' and 'clock' columns are in the correct format
filtered_data1['Date'] = pd.to_datetime(filtered_data1['Date'])
filtered_data1['clock'] = pd.to_datetime(filtered_data1['clock'], format='%H:%M:%S').dt.time

# Define the target time
target_time = pd.to_datetime('13:20:00', format='%H:%M:%S').time()

# Group by 'Date' and check if 'clock' column contains the target time
dates_with_target_time = filtered_data1.groupby('Date').apply(lambda x: target_time in x['clock'].values)

# Filter the dates where the target time exists
dates_with_target_time = dates_with_target_time[dates_with_target_time].index

# Output the dates
print(dates_with_target_time)


# In[19]:


# Group by 'Date' and check if 'clock' column contains the target time
grouped = filtered_data1.groupby('Date')

# Find dates where target time does not exist
dates_without_target_time = grouped.apply(lambda x: target_time not in x['clock'].values).index

# Function to find the closest time to target_time
def find_closest_time(times, target):
    times = sorted(times)
    closest_time = min(times, key=lambda x: abs(pd.Timestamp.combine(pd.Timestamp.now().date(), x) - pd.Timestamp.combine(pd.Timestamp.now().date(), target)))
    return closest_time

# Create a list to store the results
results = []

for date, group in grouped:
    if target_time in group['clock'].values:
        closest_time = target_time
    else:
        closest_time = find_closest_time(group['clock'].values, target_time)
    
    weather_cond = group.loc[group['clock'] == closest_time, 'weather'].values[0]
    results.append({'Date': date, 'weather_cond': weather_cond})

# Create a new DataFrame with the results
weather_cond_df = pd.DataFrame(results)

# Display the new DataFrame
print(weather_cond_df)


# In[20]:


# Assuming daily_weather3 has the 'Date' column in datetime format
daily_weather3['Date'] = pd.to_datetime(daily_weather3['Date'])

# Merge the two DataFrames on 'Date'
daily_weather3 = daily_weather3.merge(weather_cond_df, on='Date', how='left')

# Display the updated daily_weather3 DataFrame
print(daily_weather3)


# In[21]:


# Extract all unique values from the 'Weather' column
unique_weather_conditions = daily_weather3['weather_cond'].unique()

# Print the unique weather conditions
print(unique_weather_conditions)


# In[22]:


import seaborn as sns
weather_counts =daily_weather3.weather_cond.value_counts()
plt.figure(figsize=(16,5))
sns.barplot(weather_counts.index, weather_counts.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('Weather Status')
plt.xlabel('Status')
plt.ylabel('Number Of Repetition')
plt.show() # WHAT THE HECK! Let's reduce this amount of redundant information


# In[23]:


import re
# Function to clean weather descriptions
#deleting the rain/snow (precipitations in general) information if it exists
def clean_weather(description):
    if pd.isna(description):
        return description
    # Split at the first period
    parts = description.split('.', 1)
    if len(parts) > 1 and parts[1].strip():  # Check if the second part is non-empty
        return parts[1].strip()  # Return the part after the first period without adding a period
    else:
        return description  # Return the original description if no useful part after the period
daily_weather3['weather_cond'] = daily_weather3['weather_cond'].apply(clean_weather)
print(daily_weather3)    


# In[24]:


#narrowing down categories
def reduce_categories(weather):
   
    weather.weather_cond = weather.weather_cond.map({
        'Ice fog.':'Fog',
        'Haze.':'Fog',
        'Fog.':'Fog',
        'Clear.':'Sunny',
        'Sunny.':'Sunny',
        'Broken clouds.':'Scattered clouds',
        'Scattered clouds.':'Scattered clouds',
        'Overcast.':'Cloudy',
        'More clouds than sun.':'Cloudy',
        'More sun than clouds.':'Sunny',
        'Low clouds.':'Cloudy',
        'Mostly cloudy.':'Cloudy',
        'Cloudy.':'Cloudy',
        'Passing clouds.':'Passing clouds',
        'Partly sunny.':'Partly sunny',
        'Mostly sunny.':'Sunny'
    },na_action='ignore')
    return weather
daily_weather4=daily_weather3.copy()
daily_weather4 = reduce_categories(daily_weather4)


# In[25]:


daily_weather4


# In[26]:


from matplotlib import pyplot as plt
import seaborn as sns
daily_weather4.weather_cond.value_counts()
weather_counts = daily_weather4.weather_cond.value_counts()

plt.figure(figsize=(12,6))
sns.barplot(weather_counts.index, weather_counts.values, alpha=0.8)
plt.xticks(rotation=33)
plt.title('Weather Status')
plt.xlabel('Status')
plt.ylabel('Number Of Repetition')
plt.show()


# In[27]:


# Drop the 'date' column
final_data = daily_weather4.drop(columns=['Date'])

# Plot histograms for all columns except 'date'
final_data.hist(figsize=(16,12))  # You can adjust the number of bins if needed
plt.show()


# In[28]:


#splitting data into train data and test data
from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split(final_data, test_size=0.2, 
                                                   random_state=42) 
tsc= train_set.copy() 
tsc.describe() 


# In[29]:


#discovering correlations between features
tsc.corr()


# In[30]:


from pandas.plotting import scatter_matrix 
scatter_matrix(tsc, figsize=(8,6), alpha=0.4) 
plt.show()


# In[31]:


daily_weather4.head()


# In[32]:



# Assuming your DataFrame is named df
daily_weather4.to_csv('your_dataframe.csv', index=False)


# In[33]:


print(daily_weather4.dtypes)


# In[34]:


# Convert date columns to datetime
daily_weather4['Date'] = pd.to_datetime(daily_weather4['Date'])
# Convert 'Sunrise' and 'Sunset' columns to datetime, including milliseconds
daily_weather4['Sunrise'] = pd.to_datetime(daily_weather4['Sunrise'], format='%H:%M:%S.%f', errors='coerce').dt.time
daily_weather4['Sunset'] = pd.to_datetime(daily_weather4['Sunset'], format='%H:%M:%S.%f', errors='coerce').dt.time



# In[35]:


# List of columns to check
columns_to_check = ['Date', 'temp', 'wind', 'humidity', 'barometer', 'Sunrise', 'Sunset', 'Daily Production', 'weather_cond']

# Iterate over each column and print the number of missing values
for column in columns_to_check:
    missing_values_count = daily_weather4[column].isna().sum()
    print(f"The number of missing values for column '{column}' is {missing_values_count}.")

    


# In[37]:


df_cleaned = daily_weather4.dropna(subset=['humidity', 'weather_cond'])


# In[38]:


columns_to_check = ['Date', 'temp', 'wind', 'humidity', 'barometer', 'Sunrise', 'Sunset', 'Daily Production', 'weather_cond']

# Iterate over each column and print the number of missing values
for column in columns_to_check:
    missing_values_count =df_cleaned[column].isna().sum()
    print(f"The number of missing values for column '{column}' is {missing_values_count}.")


# In[39]:


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
# Split the data into features and target
X = df_cleaned.drop(columns=['Daily Production', 'Date','Sunrise','Sunset'])
y = df_cleaned['Daily Production']

# Preprocessing pipelines for numeric and categorical features
numeric_features = ['temp', 'wind', 'humidity', 'barometer']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['weather_cond']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[40]:


from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


print(df_cleaned.dtypes)


# In[42]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}


param_grids = {
    'RandomForest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20]
    },
    'LinearRegression': {},
    'DecisionTree': {
        'regressor__max_depth': [None, 10, 20]
    },
    'GradientBoosting': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1]
    },
    'XGBoost': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 6, 10]
    }
}


# In[43]:



def perform_grid_search(model, param_grid):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Dictionary to store the models
best_models = {}

# Perform Grid Search for each model and evaluate
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    best_model = perform_grid_search(model, param_grids[model_name])
    best_models[model_name] = best_model
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    

    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


# In[45]:



xgboost_model = best_models['XGBoost']
y_pred_xgboost = xgboost_model.predict(X_test)


plt.figure(figsize=(18, 10))
subset = 100
plt.plot(y_test[:subset].values, label='Actual', marker='o')
plt.plot(y_pred_xgboost[:subset], label='Predicted (XGBoost)', marker='x')
plt.xlabel('Samples')
plt.ylabel('Daily Production (kWh)')
plt.title('Actual vs Predicted Daily Production (Subset) - XGBoost')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming y_test and y_pred_xgboost are already defined

# Define the equations for the green and blue lines
def segment1_func(x):
    return 5 + x

def segment2_func(x):
    return x - 5

# Calculate the number of points lying between the green and blue segments
between_segments = np.sum((y_pred_xgboost >= segment2_func(y_test)) & (y_pred_xgboost <= segment1_func(y_test)))

# Calculate the percentage
percentage_between_segments = (between_segments / len(y_test)) * 100

print(f"Percentage of points between the green and blue segments: {percentage_between_segments:.2f}%")

# Compute min and max for the segments to cover the entire data range
x_min, x_max = y_test.min(), y_test.max()

# Define the segments
segment1_y = [5 + x_min, 5 + x_max]  # y = x + 5
segment2_y = [x_min - 5, x_max - 5]  # y = x - 5

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgboost, alpha=0.5, label='Predicted vs Actual')
plt.plot([x_min, x_max], [x_min, x_max], '--r', linewidth=2, label='Ideal Fit')
plt.plot([x_min, x_max], segment1_y, '--g', linewidth=2, label='Segment 1')
plt.plot([x_min, x_max], segment2_y, '--b', linewidth=2, label='Segment 2')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Daily Production')
plt.legend()
plt.grid(True)
plt.show()

