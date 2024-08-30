import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# TASK 1 !!!!!!!
# Load the data
file_path = "C:/Users/mauro/Downloads/archive/Retail_Transaction_Dataset.csv"
df = pd.read_csv(file_path)

# Ensure the data has the proper format
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
#print(df.head())

# Create a copy of the DataFrame for modifications (CREATIVITY PART)
dfs = df.copy()

# Sort the dataset by ID and TransactionDate
df = df.sort_values(by=['CustomerID', 'TransactionDate'])

# Function to calculate the amount for the last 3 months
def calculate_total_last_3_months(df):
    df['TotalAmountLast3Months'] = df.groupby('CustomerID')['TotalAmount'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    return df

df = calculate_total_last_3_months(df)

#print(df.head())

# The average transaction amount
df['AvgTransactionAmount'] = df.groupby('CustomerID')['TotalAmount'].transform('mean')
#print(df.head())

# The nr of distinct product categories purchased
df['DistinctCategories'] = df.groupby('CustomerID')['ProductCategory'].transform('nunique')
#print(df.head())

# The threshold to define high-value
threshold = 50

# Function to calculate monthly spending for each customer
def calculate_monthly_spending(df):
    df['YearMonth'] = df['TransactionDate'].dt.to_period('M')
    monthly_spending = df.groupby(['CustomerID', 'YearMonth'])['TotalAmount'].sum().reset_index()
    monthly_spending.rename(columns={'TotalAmount': 'MonthlyTotalAmount'}, inplace=True)
    return monthly_spending
monthly_spending = calculate_monthly_spending(df)

# Shift monthly spending (we must know if next month they will be HighValue)
monthly_spending['NextMonthTotalAmount'] = monthly_spending.groupby('CustomerID')['MonthlyTotalAmount'].shift(-1)

# Check if they customer will be HighValue
monthly_spending['IsHighValueNextMonth'] = monthly_spending['NextMonthTotalAmount'] > threshold

# Drop rows for NaN values
monthly_spending = monthly_spending.dropna(subset=['IsHighValueNextMonth'])

# Merge with main DataFrame
df = pd.merge(df, monthly_spending[['CustomerID', 'YearMonth', 'IsHighValueNextMonth']], on=['CustomerID', 'YearMonth'], how='left')

# Clean the dataset (if needed)
df = df.dropna(subset=['IsHighValueNextMonth'])
print(df.head())

# TASK 2 !!!!!!!!!

# Define features and target variable
X = df[['TotalAmountLast3Months', 'AvgTransactionAmount', 'DistinctCategories']]
y = df['IsHighValueNextMonth']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Choose the model and train it
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the metrics with 3 digit accuracy
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')


# TASK 3: Add the Feature: CustomerEngagementScore
"""The CustomerEngagementScore feature measures the level of customer engagement with various product categories and 
promotional activities over different seasons. By integrating this score, retailers can better understand which customers 
are highly engaged and tailor their strategies accordingly. The primary benefits include:

 - Targeted Marketing Efforts: Develop personalized marketing campaigns and promotions based on the engagement levels of 
customers. This targeted approach can increase the effectiveness of marketing strategies and improve customer conversion rates.

 - Optimized Resource Allocation: Allocate marketing resources and budget more efficiently by focusing on customers with 
higher engagement scores. This ensures that investments are directed towards customers who show the most potential for increased value.

 - Enhanced Customer Experience: Customize customer interactions and offers to those who exhibit higher engagement. 
This personalization can enhance the overall customer experience, leading to greater satisfaction, loyalty, and potentially increased sales."""


# Function to get the season
# 1 - Spring, 2 - Summer, 3 - Autumn, 4 - Winter
def get_season(month):
    if month in [12, 1, 2]:
        return 4  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Autumn

# Extracting the month
dfs['Season'] = dfs['TransactionDate'].dt.month.apply(get_season)

# Calculate Total Spending per Season
seasonal_spending = dfs.groupby(['CustomerID', 'Season'])['TotalAmount'].sum().reset_index()
seasonal_spending.rename(columns={'TotalAmount': 'SeasonalSpending'}, inplace=True)

# Calculate Average Seasonal Spending
average_seasonal_spending = seasonal_spending.groupby('Season')['SeasonalSpending'].mean().reset_index()
seasonal_spending = pd.merge(seasonal_spending, average_seasonal_spending, on='Season', suffixes=('', '_Avg'))
seasonal_spending['SeasonalPreferenceScore'] = seasonal_spending['SeasonalSpending'] / seasonal_spending['SeasonalSpending_Avg']

# Aggregate Seasonal Preference Score for each customer
engagement_score = seasonal_spending.groupby('CustomerID')['SeasonalPreferenceScore'].mean().reset_index()

# Merge the new feature into the DataFrame
df = pd.merge(df, engagement_score, on='CustomerID', how='left')

# Define updated features and target variable
X_updated = df[['TotalAmountLast3Months', 'AvgTransactionAmount', 'DistinctCategories', 'SeasonalPreferenceScore']]
y_updated = df['IsHighValueNextMonth']

# Split the data into training and testing sets
X_train_updated, X_test_updated, y_train_updated, y_test_updated = train_test_split(X_updated, y_updated, test_size=0.3, random_state=42)

# Initialize and train the updated Logistic Regression model
model_updated = LogisticRegression()
model_updated.fit(X_train_updated, y_train_updated)

# Make predictions on the updated test set
y_pred_updated = model_updated.predict(X_test_updated)

# Evaluate the updated model
accuracy_updated = accuracy_score(y_test_updated, y_pred_updated)
precision_updated = precision_score(y_test_updated, y_pred_updated)
recall_updated = recall_score(y_test_updated, y_pred_updated)

print(f'\nUpdated Model Accuracy: {accuracy_updated:.3f}')
print(f'Updated Model Precision: {precision_updated:.3f}')
print(f'Updated Model Recall: {recall_updated:.3f}')



