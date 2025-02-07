# Titanic Regression Challenge 2: Pre-processing the Raw Titanic Dataset

## Overview

In this challenge, I worked on pre-processing the Titanic dataset, focusing on handling missing data, feature engineering, and preparing the dataset for regression analysis. The goal was to clean the data, handle missing values effectively, and create useful features to improve the predictive modeling process.

## Objectives

- **Handle Missing Data**: I identified and removed columns with too many missing values and handled the missing 'Age' data by imputing values based on relevant subsets of the data.
- **Feature Engineering**: I extracted titles from the 'Name' column, grouped less common titles, and encoded the 'Title' column as dummy variables for modeling.

## Table of Contents

- [Installation](#installation)
- [Data Pre-processing](#data-pre-processing)
  - [Handling Missing Data](#handling-missing-data)
  - [Feature Engineering](#feature-engineering)
- [Functions](#functions)
  - [drop_columns](#drop_columns)
  - [conditional_impute](#conditional_impute)
  - [extract_title](#extract_title)
  - [group_titles](#group_titles)
  - [dummy_encode_titles](#dummy_encode_titles)
- [Example Outputs](#example-outputs)

## Installation

Before running this challenge, make sure to have the following libraries installed:

```bash
pip install pandas numpy seaborn matplotlib
```

## Data Pre-processing

### Handling Missing Data

I started by cleaning up the dataset to handle missing values.

1. **Removing Columns with Too Many Missing Values**:  
   I used a custom function to remove columns that had more than a specified percentage of missing values.

2. **Imputing 'Age' Data**:  
   For the missing 'Age' values, I used the mean or median age based on the combination of 'Sex' and 'Pclass'. This helps to preserve the relationships within the data, which could be important for regression modeling.

### Feature Engineering

1. **Extracting Titles from 'Name'**:  
   I extracted titles like 'Mr.', 'Mrs.', 'Miss.', etc., from the 'Name' column and added them as a new feature.

2. **Grouping Uncommon Titles**:  
   I grouped less common titles into a new category, 'Uncommon'. This was done to simplify the dataset and focus on the most frequent titles for modeling.

3. **Encoding Titles as Dummy Variables**:  
   I used one-hot encoding to turn the 'Title' column into dummy variables so that it could be used in machine learning models.

## Functions

### `drop_columns`

This function removes columns that have too many missing values or too few unique values.

```python
def drop_columns(input_df, threshold, unique_value_threshold):
    """
    Removes columns with too many missing values or too few unique values.
    
    Parameters:
    - input_df: pandas DataFrame
    - threshold: float, percentage of missing values allowed
    - unique_value_threshold: float, percentage of unique values allowed
    
    Returns:
    - pandas DataFrame with cleaned columns
    """
    # Remove columns with too many missing values
    missing_percentage = input_df.isnull().mean() * 100
    drop_missing = missing_percentage[missing_percentage > threshold].index
    input_df = input_df.drop(columns=drop_missing)
    
    # Remove columns with too few unique values
    unique_percentage = input_df.nunique() / len(input_df) * 100
    drop_unique = unique_percentage[unique_percentage < unique_value_threshold].index
    input_df = input_df.drop(columns=drop_unique)
    
    return input_df
```

### `conditional_impute`

I used this function to impute missing 'Age' values based on the mean or median for each 'Sex' and 'Pclass' group.

```python
def conditional_impute(input_df, choice='median'):
    """
    Imputes missing 'Age' values based on the mean or median of relevant categories.
    
    Parameters:
    - input_df: pandas DataFrame
    - choice: 'mean' or 'median', specifies the imputation strategy
    
    Returns:
    - pandas DataFrame with imputed 'Age' values
    """
    if choice not in ['mean', 'median']:
        raise ValueError("Choice must be 'mean' or 'median'")
    
    impute_values = input_df.groupby(['Sex', 'Pclass'])['Age'].agg(choice).reset_index()
    input_df = input_df.merge(impute_values, on=['Sex', 'Pclass'], how='left', suffixes=('', '_imputed'))
    input_df['Age'] = input_df['Age'].fillna(input_df['Age_imputed'])
    input_df = input_df.drop(columns=['Age_imputed'])
    
    return input_df
```

### `extract_title`

This function extracts titles from the 'Name' column and adds them as a new 'Title' column.

```python
def extract_title(input_df):
    """
    Extracts titles from the 'Name' column and adds them as a new 'Title' column.
    
    Parameters:
    - input_df: pandas DataFrame
    
    Returns:
    - pandas DataFrame with added 'Title' column
    """
    input_df['Title'] = input_df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    return input_df
```

### `group_titles`

This function groups uncommon titles into a single "Uncommon" category.

```python
def group_titles(input_df, uncommon_titles):
    """
    Groups uncommon titles into a single 'Uncommon' category.
    
    Parameters:
    - input_df: pandas DataFrame
    - uncommon_titles: list of strings, titles to group into 'Uncommon'
    
    Returns:
    - pandas DataFrame with grouped titles
    """
    input_df['Title'] = input_df['Title'].apply(lambda x: x if x not in uncommon_titles else 'Uncommon')
    return input_df
```

### `dummy_encode_titles`

This function encodes the 'Title' column into dummy variables.

```python
def dummy_encode_titles(input_df):
    """
    Encodes the 'Title' column into dummy variables.
    
    Parameters:
    - input_df: pandas DataFrame
    
    Returns:
    - pandas DataFrame with dummy-encoded 'Title' column
    """
    return pd.get_dummies(input_df, columns=['Title'], drop_first=True)
```

## Example Outputs

### `drop_columns`
```python
drop_columns(train_df, 60, 0.5).columns
# Output: ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare']
```

### `conditional_impute`
```python
conditional_impute(train_df, choice='median')[['Name', 'Age']].tail()
# Output: Imputed Age values based on Sex and Pclass categories.
```

### `extract_title`
```python
extract_title(train_df)['Title'].unique()
# Output: ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 'Rev.', 'Dr.', 'Mme.', ...]
```

### `group_titles`
```python
group_titles(title_df, uncommon_titles=['Don.', 'Rev.', 'Mme.', 'Major.', 'Sir.'])['Title'].unique()
# Output: ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Uncommon', 'Dr.', 'Ms.']
```

### `dummy_encode_titles`
```python
dummy_encode_titles(title_regrouped_df).head()
# Output: DataFrame with dummy variables for the 'Title' column.
```

## Conclusion

In this project, I focused on pre-processing the Titanic dataset, addressing missing values, extracting useful features, and preparing the data for regression modeling. These steps are crucial for building a robust machine learning model, and by cleaning and transforming the data effectively, I ensured that it was ready for analysis.
```
