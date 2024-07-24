#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import matplotlib.pyplot as plt
# Load the power production dataset
file_path = r"C:\Users\najar\Downloads\PV_Elec_Gas3.csv"
data = pd.read_csv(file_path)


# In[81]:


# Load the weather dataset
weather_file_path = r"C:\Users\najar\Downloads\weather_in_Antwerp.csv\weather_in_Antwerp.csv"
weather_data = pd.read_csv(weather_file_path, delimiter=';')


# In[82]:


weather_data


# In[83]:


#creating a new feature that seems useful for plotting
data['Daily Production'] = 0  # or any other initial value

data['Daily Production'] = data['Cumulative_solar_power'].diff()


# In[84]:


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


# In[85]:


data


# In[86]:


duplicated_dataset = weather_data.copy()
# Create a 'Date' column by combining 'year', 'month', and 'day'
duplicated_dataset['Date'] = pd.to_datetime(duplicated_dataset[['year', 'month', 'day']])

# Drop the original 'year', 'month', and 'day' columns
duplicated_dataset= duplicated_dataset.drop(columns=['year', 'month', 'day'])


# In[87]:


duplicated_dataset.drop(columns={'Unnamed: 0'})


# In[88]:


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


# In[55]:


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


# In[89]:


# Group by date and calculate daily means
daily_weather =filtered_data.groupby('Date').agg({
    'temp': 'mean',
    'wind': 'mean',
    'humidity': 'mean',
    'barometer': 'mean',
    'Sunrise': 'first',  # Keep the first occurrence of Sunrise time for each date
    'Sunset': 'first'
    
}).reset_index()


# In[90]:


daily_weather


# In[91]:


daily_weather2=daily_weather.copy()


# In[92]:


data


# In[93]:


# Merge the dataframes on the 'date' and 'Date' columns using an inner join
merged_data = pd.merge(daily_weather2, data[['date', 'Daily Production']], left_on='Date', right_on='date', how='inner')

# Assign the 'Daily Production' column from merged_data to daily_weather2
daily_weather2 = merged_data.drop(columns='date')

# Check the result
print(daily_weather2.head())


# In[94]:


daily_weather2


# In[95]:


daily_weather3=daily_weather2.copy()


# In[96]:


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


# In[97]:


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


# In[98]:


# Assuming daily_weather3 has the 'Date' column in datetime format
daily_weather3['Date'] = pd.to_datetime(daily_weather3['Date'])

# Merge the two DataFrames on 'Date'
daily_weather3 = daily_weather3.merge(weather_cond_df, on='Date', how='left')

# Display the updated daily_weather3 DataFrame
print(daily_weather3)


# In[99]:


# Extract all unique values from the 'Weather' column
unique_weather_conditions = daily_weather3['weather_cond'].unique()

# Print the unique weather conditions
print(unique_weather_conditions)


# In[107]:


import seaborn as sns
weather_counts =daily_weather3.weather_cond.value_counts()
plt.figure(figsize=(16,5))
sns.barplot(weather_counts.index, weather_counts.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('Weather Status')
plt.xlabel('Status')
plt.ylabel('Number Of Repetition')
plt.show() # WHAT THE HECK! Let's reduce this amount of redundant information


# In[108]:


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


# In[109]:


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


# In[110]:


daily_weather4


# In[111]:


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


# In[114]:


# Drop the 'date' column
final_data = daily_weather4.drop(columns=['Date'])

# Plot histograms for all columns except 'date'
final_data.hist(figsize=(16,12))  # You can adjust the number of bins if needed
plt.show()


# In[115]:


from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split(final_data, test_size=0.2, 
                                                   random_state=42) 
tsc= train_set.copy() 
tsc.describe() 


# In[116]:


#discovering correlations between features
tsc.corr()


# In[117]:


from pandas.plotting import scatter_matrix 
scatter_matrix(tsc, figsize=(16,18), alpha=0.4) 
plt.show()


# In[118]:


daily_weather4.head()


# In[ ]:




