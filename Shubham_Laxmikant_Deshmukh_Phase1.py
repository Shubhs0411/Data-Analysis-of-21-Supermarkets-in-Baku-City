#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===================================#
# Name: Shubham Laxmikant Deshmukh   #
# ===================================#


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# %%

# Reading the dataset from the csv file
df = pd.read_csv("/Users/shubhamlaxmikantdeshmukh/Downloads/esas_mehsullar 2.csv")

# taking limited records
df = df.head(120000)

# Renaming column Names
df = df.rename(
    columns={'Unnamed: 0': 'ID', 'satish_kodu': 'Receipt_Number', 'mehsul_kodu': 'Product_Code', 'mehsul_ad': 'Product',
             'mehsul_kateqoriya': 'Category', 'mehsul_qiymet': 'Price', 'satish_tarixi': 'Date',
             'endirim_kompaniya': 'Discount', 'bonus_kart': 'Bonus_cart', 'magaza_ad': 'Store_Branch',
             'magaza_lat': 'Store_Lat', 'magaza_long': 'Store_Long'})

# Printing the first 5 rows from the dataframe
print(df.head())

# Printing the info of df
print(df.info())

# Checking if there are any na values in df
print(df.isna().sum())

# droping the na values in df and checking the if they are gone
clean_df = df.dropna()

# dropping unescessay columns
clean_df = clean_df.drop('ID', axis=1)

# reseting index
clean_df = clean_df.reset_index(drop=True)

# checking for duplicate rows and dropping them
duplicated_rows = clean_df.duplicated()
clean_df = clean_df.drop_duplicates()

# Coverting type of some coulmns
clean_df = clean_df.copy()  # Create a copy of the DataFrame
clean_df['Product'] = clean_df['Product'].astype('category')
clean_df.loc[:, 'Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
clean_df['Category'] = clean_df['Category'].astype('category')
clean_df['Discount'] = clean_df['Discount'].astype('category')
clean_df['Store_Branch'] = clean_df['Store_Branch'].astype('category')

# printing the info of clean_df and its first 10 rows
print(clean_df.info())
print(clean_df.head(5))

# In[2]:


# Replace multiple values in the 'Discount' column
replace_dict = {'Sərin Yay günləri': 'Summer Season', 'S?rf?li Yaz': 'Fifth of May Season',
                'Bərəkətli Novruz': 'Happy Nowruz Season', 'Yeni il fürsətləri': 'New Year Season',
                'Payız endirimləri': 'Autumn Season'}
clean_df['Discount'] = clean_df['Discount'].replace(replace_dict)

# Translate values in the 'Category' and 'Store Branch' column
from googletrans import Translator

translator = Translator()
clean_df['Category'] = clean_df['Category'].apply(lambda x: translator.translate(x).text)
clean_df['Store_Branch'] = clean_df['Store_Branch'].apply(lambda x: translator.translate(x).text)

# In[3]:


print(clean_df.head(5))
print(clean_df.isna().sum())

# In[4]:


# Unique values and counts for 'Product'
Receipt_Number_counts = clean_df['Receipt_Number'].value_counts()

# Unique values and counts for 'Product'
product_counts = clean_df['Product'].value_counts()

# Unique values and counts for 'Discount'
discount_counts = clean_df['Discount'].value_counts()

# Unique values and counts for 'Store_Branch'
store_branch_counts = clean_df['Store_Branch'].value_counts()

# Unique values and counts for 'Category'
category_counts = clean_df['Category'].value_counts()

# Displaying the results
print("\nUnique values and counts for 'Receipt_Number':")
print(Receipt_Number_counts)

print("\nUnique values and counts for 'Product':")
print(product_counts)

print("\nUnique values and counts for 'Discount':")
print(discount_counts)

print("\nUnique values and counts for 'Store_Branch':")
print(store_branch_counts, len(store_branch_counts))

print("\nUnique values and counts for 'Category':")
print(category_counts)

# # Line plot

# In[5]:


# Reading the dataset from the csv file
sub_df = pd.read_csv("/Users/shubhamlaxmikantdeshmukh/Downloads/esas_mehsullar 2.csv")

# #taking limited records
# df=df.head(120000)

# Renaming column Names
sub_df = sub_df.rename(
    columns={'Unnamed: 0': 'ID', 'satish_kodu': 'Receipt_Number', 'mehsul_kodu': 'Product_Code', 'mehsul_ad': 'Product',
             'mehsul_kateqoriya': 'Category', 'mehsul_qiymet': 'Price', 'satish_tarixi': 'Date',
             'endirim_kompaniya': 'Discount', 'bonus_kart': 'Bonus_cart', 'magaza_ad': 'Store_Branch',
             'magaza_lat': 'Store_Lat', 'magaza_long': 'Store_Long'})

# droping the na values in df and checking the if they are gone
sub_df = sub_df.dropna()

# dropping unescessay columns
sub_df = sub_df.drop('ID', axis=1)

# reseting index
sub_df = sub_df.reset_index(drop=True)

# checking for duplicate rows and dropping them
duplicated_rows = clean_df.duplicated()
clean_df = clean_df.drop_duplicates()

# Coverting type of some coulmns
sub_df = sub_df.copy()  # Create a copy of the DataFrame
sub_df['Product'] = sub_df['Product'].astype('category')
sub_df.loc[:, 'Date'] = pd.to_datetime(sub_df['Date'], dayfirst=True)
sub_df['Category'] = sub_df['Category'].astype('category')
sub_df['Discount'] = sub_df['Discount'].astype('category')
sub_df['Store_Branch'] = sub_df['Store_Branch'].astype('category')

new_df = sub_df[['Date', 'Price']].copy()

new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
unique_dates_count = new_df['Date'].nunique()
print("Number of unique dates:", unique_dates_count)

import seaborn as sns
import matplotlib.pyplot as plt

# Sort the DataFrame by 'Date'
new_df.sort_values(by='Date', inplace=True)

# Plotting with Seaborn
plt.figure(figsize=(24, 14))
sns.lineplot(x='Date', y='Price', data=new_df, marker='o', color='b')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Prices Over Time')

# Show the plot
plt.show()

# # Bar plot

# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with counts of Bonus Cart usage by category
bonus_cart_by_category = clean_df.groupby(['Category', 'Bonus_cart']).size().unstack()

# Plotting
plt.figure(figsize=(24, 8))
bonus_cart_by_category.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0, 1200)
plt.title('Stacked Bar Plot of Bonus Cart Usage by Category')
plt.xticks(rotation=60)
plt.legend(title='Bonus Cart')
plt.gcf().set_size_inches(18, 12)
plt.tight_layout()
plt.show()

# Create a DataFrame with counts of discounts by category
discount_by_category = clean_df.groupby(['Category', 'Discount']).size().unstack()

# Plotting
plt.figure(figsize=(24, 8))
discount_by_category.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0, 14000)
plt.title('Stacked Bar Plot of Discount Distribution by Category')
plt.xticks(rotation=60)
plt.gcf().set_size_inches(18, 12)
plt.legend(title='Discount')
plt.show()

# Create a DataFrame with counts of Bonus Cart usage by store branch
bonus_cart_by_branch = clean_df.groupby(['Store_Branch', 'Bonus_cart']).size().unstack()

# Plotting
plt.figure(figsize=(24, 8))
bonus_cart_by_branch.plot(kind='bar', stacked=True)
plt.xlabel('Store Branch')
plt.ylabel('Count')
plt.title('Stacked Bar Plot of Bonus Cart Usage by Store Branch')
plt.xticks(rotation=90)
plt.legend(title='Bonus Cart')
plt.show()

# Stacked Bar Plot for Category by Store_Branch
category_by_branch = clean_df.groupby(['Category', 'Store_Branch']).size().unstack()
category_by_branch.plot(kind='bar', stacked=True, figsize=(20, 12), colormap='viridis')
plt.title('Category Distribution by Store Branch', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Store Branch', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})

plt.xticks(rotation=90)
plt.show()

# Stacked Bar Plot for Bonus_cart by Store_Branch
bonus_cart_by_branch = clean_df.groupby(['Bonus_cart', 'Store_Branch']).size().unstack()
bonus_cart_by_branch.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set2')
plt.title('Bonus Cart Usage Distribution by Store Branch', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Store Branch', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.xticks(rotation=45)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of seaborn
sns.set(style="whitegrid")

# Set up the matplotlib figure with a larger size
plt.figure(figsize=(16, 10))

# Create a grouped bar plot
sns.barplot(x='Discount', y='Price', hue='Category', data=clean_df, ci=None)

# Adjust legend size and position
plt.legend(title='Category', title_fontsize='14', loc='upper right', fontsize='12')

# Add labels and title
plt.xlabel('Discount', fontsize=14)
plt.ylabel('Average Price', fontsize=14)
plt.title('Grouped Bar Plot of Average Price by Discount and Category', fontsize=16)

# Show the plot
plt.show()

# # Count plot
#

# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the count of each receipt number
receipt_number_counts = clean_df['Receipt_Number'].value_counts()

# Filter receipt numbers with count > 2
selected_receipt_numbers = receipt_number_counts[receipt_number_counts > 2].index

# Filter the dataframe based on selected receipt numbers
selected_df = clean_df[clean_df['Receipt_Number'].isin(selected_receipt_numbers)]

# Create a count plot for store branches
plt.figure(figsize=(12, 8))
sns.countplot(x='Store_Branch', data=selected_df)
plt.title("Store Branches for Receipt Numbers with Count > 2")
plt.xlabel("Store Branch")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)

# Find the store with the maximum count
most_common_store = selected_df['Store_Branch'].mode().iloc[0]
max_count = selected_df['Store_Branch'].value_counts().max()

# Highlight the store with the most counts
ax = plt.gca()
for p in ax.patches:
    if p.get_height() == max_count:
        p.set_facecolor('red')  # Highlight the bar with the most counts

plt.show()

# Print the name of the store with the most counts
print(f"The store with the most counts is: {most_common_store} (Count: {max_count})")

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the count of each receipt number
receipt_number_counts = clean_df['Receipt_Number'].value_counts()

# Filter receipt numbers with count > 7
selected_receipt_numbers = receipt_number_counts[receipt_number_counts > 7].index

# Filter the dataframe based on selected receipt numbers
selected_df = clean_df[clean_df['Receipt_Number'].isin(selected_receipt_numbers)]

# Create a count plot for categories
plt.figure(figsize=(20, 8))
sns.countplot(x='Category', data=selected_df)
plt.title("Categories for Receipt Numbers with Count > 7")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.gcf().set_size_inches(28, 12)

# Find the category with the maximum count
most_common_category = selected_df['Category'].mode().iloc[0]
max_count = selected_df['Category'].value_counts().max()

# Highlight the category with the most counts
ax = plt.gca()
for p in ax.patches:
    if p.get_height() == max_count:
        p.set_facecolor('red')  # Highlight the bar with the most counts

plt.show()

# Print the name of the category with the most counts
print(f"The category with the most counts is: {most_common_category} (Count: {max_count})")

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the counts for each product
product_counts = clean_df['Product'].value_counts()

# Identify products with counts less than 12
products_to_replace = product_counts[product_counts < 12].index

# Replace those products with 'Others'
clean_df['Product'] = clean_df['Product'].replace(products_to_replace, 'Others')

# Filter out 'Others' before creating the count plot
filtered_df = clean_df[clean_df['Product'] != 'Others']

# Create a count plot with descending order
plt.figure(figsize=(12, 8))
sns.countplot(x='Product', data=filtered_df, order=filtered_df['Product'].value_counts().index[::-1])
plt.title("Count of Products (Counts > 11)")
plt.xlabel("Product")
plt.ylabel("Count")
plt.ylim(0, 40)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Create a count plot with ascending order
plt.figure(figsize=(22, 8))
sns.countplot(x='Store_Branch', data=clean_df, order=clean_df['Store_Branch'].value_counts().index[::-1])
plt.title("Number of Products Bought in a Store Branch")
plt.xlabel("Store Branch")
plt.ylabel("# of Products Bought")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Create a count plot with asc order and ylim
plt.figure(figsize=(12, 8))
sns.countplot(x='Category', data=clean_df,
              order=clean_df['Category'].value_counts().index[::-1])  # Reverse the order for descending
plt.title("Count of Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.ylim(0, 4000)  # Set y-axis limit
plt.grid(True)
plt.show()

# Count Plot for 'Discount'
plt.figure(figsize=(14, 10))
sns.countplot(x='Discount', data=clean_df, palette='pastel', order=clean_df['Discount'].value_counts().index[::-1])
plt.title('Count of Discounts', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Discount', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.show()

# Count Plot for 'Bonus_Cart'
plt.figure(figsize=(14, 10))
sns.countplot(x='Bonus_cart', data=clean_df, palette='pastel')
plt.title('Count of Bonus cart', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.xlabel('Bonus cart', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
plt.show()

# # Pie plots

# In[41]:


# Calculate the counts for each product
product_counts = clean_df['Product'].value_counts()

# Identify products with counts less than 12
products_to_replace = product_counts[product_counts < 12].index

# Replace those products with 'Others'
clean_df['Product'] = clean_df['Product'].replace(products_to_replace, 'Others')

# Remove 'Others' from the counts and labels
product_counts = product_counts[product_counts.index != 'Others']
product_labels = product_counts.index

# Create a pie plot with cool colors
cool_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf']
plt.figure(figsize=(27, 20))
plt.pie(product_counts, labels=product_labels, autopct='%1.1f%%', startangle=140, colors=cool_colors,
        wedgeprops=dict(width=0.3))
plt.title("Distribution of Products (Counts > 11)")
plt.show()

# Create a DataFrame with counts of products by discount category
discount_counts = clean_df['Discount'].value_counts()

# Plotting
plt.figure(figsize=(10, 10))
plt.pie(discount_counts, labels=discount_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Pie Chart of Discount Distribution')
plt.show()

# Create a DataFrame with counts of products by Bonus Cart usage
bonus_cart_counts = clean_df['Bonus_cart'].value_counts()

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(bonus_cart_counts, labels=bonus_cart_counts.index, autopct='%1.1f%%', startangle=90,
        colors=plt.cm.Accent.colors)
plt.title('Pie Chart of Bonus Cart Usage')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the counts for each category
category_counts = clean_df['Category'].value_counts()

# Set a threshold for percentage labels
percentage_threshold = 1.0  # Display labels only for slices with percentage greater than or equal to this threshold

# Identify slices with percentage below the threshold and combine them into 'Others' slice
below_threshold = category_counts[category_counts / category_counts.sum() * 100 < percentage_threshold]
category_counts['Others'] = below_threshold.sum()
category_counts = category_counts[category_counts / category_counts.sum() * 100 >= percentage_threshold]

# Plotting a pie chart with improved readability
plt.figure(figsize=(20, 20))
plt.pie(category_counts, labels=category_counts.index,
        autopct=lambda p: '{:.1f}%'.format(p) if p >= percentage_threshold else '', startangle=90,
        colors=plt.cm.Set3.colors, textprops={'fontsize': 20})
plt.title('Pie Chart of Category Distribution', fontsize=20)
plt.show()

# # Distplot

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting a distplot for the 'Price' column
plt.figure(figsize=(12, 6))
sns.distplot(clean_df['Price'], bins=20, hist_kws={'color': 'skyblue'}, kde_kws={'color': 'orange'})
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()

# # Heatmap

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Selecting numerical columns for correlation heatmap
numerical_columns = clean_df.select_dtypes(include=['float64', 'int64']).columns

# Calculate the correlation matrix for numerical columns
correlation_matrix = clean_df[numerical_columns].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Columns')
plt.show()

# # Pairplot

# In[11]:


# Pairplot for Numerical Columns
numerical_columns = clean_df.select_dtypes(include=['int64', 'float64'])
sns.pairplot(numerical_columns)
plt.show()

# # Histplot

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting a histogram for the 'Price' column
plt.figure(figsize=(12, 6))
sns.histplot(clean_df['Price'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# # QQ plot

# In[13]:


import statsmodels.api as sm

plt.figure(figsize=(8, 6))
sm.qqplot(clean_df["Price"], line='s', color='salmon')
plt.title("QQ-Plot for Price", fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
plt.show()

# # KDE with fill

# In[14]:


# KDE Plot with Fill, Alpha, Palette, and Linewidth
sns.kdeplot(data=clean_df, x='Price', fill=True, alpha=0.6, palette='viridis', linewidth=2)
plt.title('KDE Plot with Fill')
plt.show()

# # lm or regplot

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate mean prices for each unique date
mean_prices = new_df.groupby('Date')['Price'].mean().reset_index()

# Convert 'Date' to numeric values
mean_prices['Numeric_Date'] = mean_prices['Date'].apply(lambda x: x.toordinal())

# Create a scatter plot with regression line
plt.figure(figsize=(24, 14))
sns.lmplot(x='Numeric_Date', y='Price', data=mean_prices, scatter_kws={'s': 100}, aspect=2, height=6)

# Adding labels and title
plt.xlabel('Numeric Date')
plt.ylabel('Mean Price')
plt.title('Scatter Plot with Regression Line of Mean Prices Over Unique Dates')

# Show the plot
plt.show()

# # Multivariate Box or Boxen plot

# In[16]:


plt.figure(figsize=(32, 15))
sns.boxplot(x='Category', y='Price', hue='Discount', data=clean_df, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Price')
plt.title('Box Plot of Prices by Category and Discount')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()

# # Area plot

# In[42]:


# Area Plot
plt.figure(figsize=(32, 15))
sns.lineplot(data=clean_df, x='Date', y='Price', hue='Category', estimator='sum', ci=None)
plt.title('Area Plot Over Time')
plt.show()

# # Violin plot

# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of seaborn
sns.set(style="whitegrid")

# Set up the matplotlib figure with a larger size
plt.figure(figsize=(18, 12))

# Violin plot for Price distribution by Discount category
plt.subplot(2, 2, 1)
sns.violinplot(x='Discount', y='Price', data=clean_df, palette='viridis')
plt.xlabel('Discount', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Violin Plot of Price by Discount', fontsize=16)

# Violin plot for Price distribution by Category
plt.subplot(2, 2, 2)
sns.violinplot(x='Category', y='Price', data=clean_df, palette='muted')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Violin Plot of Price by Category', fontsize=16)
plt.xticks(rotation=45, ha='right')

# Violin plot for Price distribution by Store Branch
plt.subplot(2, 2, 3)
sns.violinplot(x='Store_Branch', y='Price', data=clean_df, palette='deep')
plt.xlabel('Store Branch', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Violin Plot of Price by Store Branch', fontsize=16)
plt.xticks(rotation=45, ha='right')

# Violin plot for Price distribution by Discount and Category without 'split'

# sns.violinplot(x='Discount', y='Price', hue='Category', data=clean_df, palette='pastel')
# plt.xlabel('Discount', fontsize=14)
# plt.ylabel('Price', fontsize=14)
# plt.title('Violin Plot of Price by Discount and Category', fontsize=16)
# Convert 'Date' column to datetime
clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
plt.subplot(2, 2, 4)
sns.violinplot(x=clean_df['Date'].dt.dayofweek, y='Price', data=clean_df, inner='quartile', palette='coolwarm')
plt.xlabel('Day of Week')
plt.ylabel('Price')
plt.title('Daily Sales Distribution')
plt.show()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# # Joinplot

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Discount' to numeric labels
clean_df['Discount_Label'] = clean_df['Discount'].astype('category').cat.codes

# Set the style of seaborn
sns.set(style="whitegrid")

# Set up the matplotlib figure with a larger size
plt.figure(figsize=(12, 8))

# Create a joint plot with KDE and scatter representation
sns.jointplot(x='Price', y='Discount_Label', data=clean_df, kind='scatter', hue='Category', height=8, palette='viridis')

# Add labels and title
plt.xlabel('Price', fontsize=14)
plt.ylabel('Discount', fontsize=14)
plt.title('Joint Plot with KDE and Scatter Representation', fontsize=16)

# Show the plot
plt.show()

# # Rugplot

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame with unique dates and mean prices
mean_prices_by_date = new_df.groupby('Date')['Price'].mean().reset_index()

# Create a rug plot for unique dates and mean prices
plt.figure(figsize=(16, 4))
rug_plot = sns.rugplot(data=mean_prices_by_date, x='Date', height=0.5, hue='Price', palette='viridis')
plt.title('Rug Plot of Mean Prices Over Unique Dates')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.yticks([])
plt.show()

# # 3D plot

# In[21]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import dates


new_df['Date'] = pd.to_datetime(new_df['Date'])

# Calculate mean prices for each unique date
mean_prices = new_df.groupby('Date')['Price'].mean().reset_index()

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert 'Date' to numerical format using matplotlib's date2num
num_dates = dates.date2num(mean_prices['Date'])

scatter = ax.scatter(num_dates, mean_prices['Price'], c=mean_prices['Price'], cmap='viridis', s=50)

# Adding labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Mean Price')
ax.set_zlabel('Mean Price')
ax.set_title('3D Scatter Plot of Mean Prices Over Time')

# Show the colorbar
fig.colorbar(scatter, ax=ax, label='Mean Price')

# Format x-axis as dates
ax.xaxis.set_major_locator(dates.YearLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))

# Show the plot
plt.show()

# # Cluster map

# In[22]:


# Selecting numerical columns for correlation heatmap
numerical_columns = clean_df.select_dtypes(include=['float64', 'int64']).columns

# Calculate the correlation matrix for numerical columns
correlation_matrix = clean_df[numerical_columns].corr()
sns.clustermap(correlation_matrix, cmap='coolwarm', linewidths=0.5, annot=True)
plt.title('Cluster Map')
plt.show()

# # Strip plot

# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate mean prices for each unique date
mean_prices = new_df.groupby('Date')['Price'].mean().reset_index()

# Create a strip plot
plt.figure(figsize=(24, 12))
strip_plot = sns.stripplot(data=mean_prices, x='Date', y='Price', jitter=True, size=5)
plt.title('Strip Plot of Mean Prices Over Unique Dates')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.xticks(rotation=90)
plt.show()

# # Swarm plot

# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate mean prices for each unique date
mean_prices = new_df.groupby('Date')['Price'].mean().reset_index()

# Create a swarm plot
plt.figure(figsize=(27, 12))
swarm_plot = sns.swarmplot(data=mean_prices, x='Date', y='Price', size=8)
plt.title('Swarm Plot of Mean Prices Over Unique Dates')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.xticks(rotation=45)
plt.show()

# # Some questions

# In[25]:


# Which Category is the costliest?

# Group by 'Category' and calculate the average price for each category
average_prices_by_category = clean_df.groupby('Category')['Price'].mean().sort_values(ascending=True)

# Plotting
plt.figure(figsize=(14, 12))
sns.barplot(x=average_prices_by_category.values, y=average_prices_by_category.index, palette='viridis')

# Adding labels and title
plt.xlabel('Average Price')
plt.ylabel('Category')
plt.title('Average Prices by Category')

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Find the costliest product and its count as compared to others
costliest_product = clean_df.groupby('Product')['Price'].mean().idxmax()

# Filter the DataFrame for the costliest product
costliest_product_data = clean_df[clean_df['Product'] == costliest_product]

# Print the name, count, and average price of the costliest product
print(f"\nCostliest Product: {costliest_product}")
print(f"Count: {len(costliest_product_data)}")
print(f"Average Price: ${costliest_product_data['Price'].mean():.2f}")

# Plotting
plt.figure(figsize=(14, 10))
count_plot = sns.countplot(x='Product', data=clean_df, order=clean_df['Product'].value_counts().index[::-1],
                           palette='viridis')

# Highlight the bar for the costliest product
costliest_product_index = clean_df['Product'].value_counts().index.get_loc(costliest_product)
count_plot.patches[costliest_product_index].set_facecolor('red')

# Adding labels and title
plt.xlabel('Product')
plt.ylabel('Count')
plt.ylim(0, 50)
plt.title(f'Count of {costliest_product} (Costliest Product)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()

# Which Store has the max sale in each season and by what count?

# Group by 'Discount' and 'Store_Branch', and calculate the count for each combination
seasonal_store_counts = clean_df.groupby(['Discount', 'Store_Branch']).size().reset_index(name='Count')

# Find the store with the maximum count in each season
max_store_in_season = seasonal_store_counts.loc[seasonal_store_counts.groupby('Discount')['Count'].idxmax()]

# Plotting
plt.figure(figsize=(14, 8))
ax = sns.barplot(x='Count', y='Store_Branch', hue='Discount', data=max_store_in_season, palette='Set2')

# Adding labels and title
plt.xlabel('Count')
plt.ylabel('Store Branch')
plt.title('Store with Maximum Sale Count in Each Season')

# Show the plot
plt.show()

# Grouping by Store Branch and calculating the average count of Receipt Numbers
average_receipt_count_by_branch = clean_df.groupby('Store_Branch')['Receipt_Number'].count().reset_index()

# Sorting the DataFrame by average count in ascending order
average_receipt_count_by_branch = average_receipt_count_by_branch.sort_values(by='Receipt_Number', ascending=True)

# Plotting a bar plot with explicit order
plt.figure(figsize=(14, 8))
sns.barplot(x='Store_Branch', y='Receipt_Number', data=average_receipt_count_by_branch,
            order=average_receipt_count_by_branch['Store_Branch'], palette='viridis')
plt.title('Average Count of Receipt Numbers by Store Branch (Ascending Order)')
plt.xlabel('Store Branch')
plt.ylabel('Average Count of Receipt Numbers')
plt.xticks(rotation=45)
plt.show()

# Convert 'Date' column to datetime with specified format
clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')

# Extract month and year from the 'Date' column
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

# Create subplots for each store branch
store_branches = clean_df['Store_Branch'].unique()

plt.figure(figsize=(35, 30))
for i, branch in enumerate(store_branches, 1):
    plt.subplot(7, 3, i)
    branch_data = clean_df[clean_df['Store_Branch'] == branch]
    monthly_prices = branch_data.groupby(['Year', 'Month'])['Price'].sum().reset_index()
    sns.lineplot(x='Month', y='Price', hue='Year', data=monthly_prices, marker='o')
    plt.title(f'Total Revenue Over Months - {branch}')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plt.legend(title='Year')

plt.tight_layout()
plt.show()

# Convert 'Date' column to datetime with specified format
clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')

# Extract month and year from the 'Date' column
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

# Find the store with the maximum total price for each month
max_price_per_month = clean_df.groupby(['Store_Branch', 'Year', 'Month'])['Price'].sum().reset_index()
idx = max_price_per_month.groupby(['Year', 'Month'])['Price'].idxmax()
max_price_per_month = max_price_per_month.loc[idx]

for _, row in max_price_per_month.iterrows():
    month_name = pd.to_datetime(f'{int(row["Year"])}-{int(row["Month"])}-1').strftime('%B')
    print(f"For {month_name}, the store with the max total price is {row['Store_Branch']} with ${row['Price']:.2f}")

# Set up subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 32))
fig.suptitle('Category Sold the Most in Each Season', fontsize=20)

# Group by 'Discount' and 'Category', and calculate the count for each combination
seasonal_category_counts = clean_df.groupby(['Discount', 'Category']).size().reset_index(name='Count')

# List of unique seasons (discounts)
seasons = seasonal_category_counts['Discount'].unique()

# Iterate over each season and create a subplot
for i, season in enumerate(seasons, 1):
    plt.subplot(3, 2, i)

    # Filter data for the current season
    season_data = seasonal_category_counts[seasonal_category_counts['Discount'] == season]

    # Sort the data for the current season in descending order
    season_data = season_data.sort_values(by='Count', ascending=False)

    # Find the category with the maximum count in the current season
    max_category = season_data.loc[season_data['Count'].idxmax(), 'Category']

    # Plotting
    sns.barplot(x='Count', y='Category', data=season_data, palette='Set3')

    # Highlight the bar for the category with the maximum count
    max_category_index = season_data['Category'].values.tolist().index(max_category)
    plt.gca().patches[max_category_index].set_facecolor('red')

    # Adding labels and title
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.xlim(0, 700)
    plt.title(f'Season: {season}')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()



# Define cool and unique colors for each store branch
branch_colors = {
    'N.Narimanov': '#1f77b4',
    'Khatai': '#ff7f0e',
    'Machine market': '#2ca02c',
    'Ahmadli': '#d62728',
    'Hypermarket': '#9467bd',
    'M. Ajami': '#8c564b',
    'Ayna Sultanova': '#e377c2',
    'Narimanov-2': '#7f7f7f',
    'C. Mammadguluzade': '#bcbd22',
    'Almond': '#17becf',
    'People\'s friendship': '#aec7e8',
    'SEK 8 million': '#ffbb78',
    'Hazi Aslanov-1': '#98df8a',
    'Zabrat': '#c5b0d5',
    'Hazi Aslanov-2': '#c49c94',
    'Yasamal': '#f7b6d2',
    'Small church': '#dbdb8d',
    'Radiozavod': '#9edae5',
    '28-May': '#d62728',
    'New Yasamal': '#1f77b4',
    '20th January': '#2ca02c'
}

# Convert 'Date' column to datetime with specified format
clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')

# Extract month and year from the 'Date' column
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

# Create a line plot for total prices over months for each store branch with custom colors
plt.figure(figsize=(24, 14))
for branch in store_branches:
    branch_data = clean_df[clean_df['Store_Branch'] == branch]
    monthly_prices = branch_data.groupby(['Year', 'Month'])['Price'].sum().reset_index()
    sns.lineplot(x='Month', y='Price', data=monthly_prices, marker='o', label=branch, color=branch_colors[branch])

# Adding labels and title with increased font size
plt.title('Total Revenue Over Months for Each Store Branch', fontsize=24)
plt.xlabel('Month', fontsize=18)
plt.xlim(0, 9)
plt.ylabel('Total Revenue', fontsize=18)
plt.legend(title='Store Branch', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)  # Increased font size

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

# Convert 'Date' column to datetime
clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')

# Extract month and year
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

# Create a line plot for total sales over months
plt.figure(figsize=(16, 8))

# Use the 'viridis' color palette
colors = sns.color_palette('viridis', n_colors=len(clean_df['Year'].unique()))

monthly_sales = clean_df.groupby(['Year', 'Month'])['Price'].sum().reset_index()
sns.lineplot(x='Month', y='Price', hue='Year', data=monthly_sales, marker='o', palette=colors)

plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend(title='Year')
plt.show()

# In[55]:


import pandas as pd
from prettytable import PrettyTable



# Convert 'Date' column to datetime with specified format
clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')

# Extract month and year from the 'Date' column
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

# Find the store with the maximum total price for each month
max_price_per_month = clean_df.groupby(['Store_Branch', 'Year', 'Month'])['Price'].sum().reset_index()
idx = max_price_per_month.groupby(['Year', 'Month'])['Price'].idxmax()
max_price_per_month = max_price_per_month.loc[idx]

# Create a PrettyTable
table = PrettyTable()
table.field_names = ["Month", "Store with Max Total Price", "Max Total Price"]

for _, row in max_price_per_month.iterrows():
    month_name = pd.to_datetime(f'{int(row["Year"])}-{int(row["Month"])}-1').strftime('%B')
    table.add_row([month_name, row['Store_Branch'], f"${row['Price']:.2f}"])

print(table)

# # PCA and normality testing

# In[26]:


# Reading the dataset from the csv file
sub_df = pd.read_csv("/Users/shubhamlaxmikantdeshmukh/Downloads/esas_mehsullar 2.csv")

# #taking limited records
# df=df.head(120000)

# Renaming column Names
sub_df = sub_df.rename(
    columns={'Unnamed: 0': 'ID', 'satish_kodu': 'Receipt_Number', 'mehsul_kodu': 'Product_Code', 'mehsul_ad': 'Product',
             'mehsul_kateqoriya': 'Category', 'mehsul_qiymet': 'Price', 'satish_tarixi': 'Date',
             'endirim_kompaniya': 'Discount', 'bonus_kart': 'Bonus_cart', 'magaza_ad': 'Store_Branch',
             'magaza_lat': 'Store_Lat', 'magaza_long': 'Store_Long'})

# droping the na values in df and checking the if they are gone
sub_df = sub_df.dropna()

# dropping unescessay columns
sub_df = sub_df.drop('ID', axis=1)

# reseting index
sub_df = sub_df.reset_index(drop=True)

# checking for duplicate rows and dropping them
duplicated_rows = clean_df.duplicated()
clean_df = clean_df.drop_duplicates()

# Coverting type of some coulmns
sub_df = sub_df.copy()  # Create a copy of the DataFrame
sub_df['Receipt_Number'] = sub_df['Receipt_Number'].astype('int')
sub_df['Product'] = sub_df['Product'].astype('category')
sub_df.loc[:, 'Date'] = pd.to_datetime(sub_df['Date'], dayfirst=True)
sub_df['Category'] = sub_df['Category'].astype('category')
sub_df['Discount'] = sub_df['Discount'].astype('category')
sub_df['Store_Branch'] = sub_df['Store_Branch'].astype('category')

# Replace multiple values in the 'Discount' column
replace_dict = {'Sərin Yay günləri': 'Summer Season', 'S?rf?li Yaz': 'Fifth of May Season',
                'Bərəkətli Novruz': 'Happy Nowruz Season', 'Yeni il fürsətləri': 'New Year Season',
                'Payız endirimləri': 'Autumn Season'}
sub_df['Discount'] = sub_df['Discount'].replace(replace_dict)

# Translate values in the 'Category' and 'Store Branch' column
from googletrans import Translator

translator = Translator()
sub_df['Category'] = sub_df['Category'].apply(lambda x: translator.translate(x).text)
sub_df['Store_Branch'] = sub_df['Store_Branch'].apply(lambda x: translator.translate(x).text)

# Q2)
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
numerical_data = sub_df.select_dtypes(include=[np.number])  # Select numerical columns only

standardized_data = scaler.fit_transform(numerical_data)
print(standardized_data)
# Calculate the correlation matrix
correlation_matrix = pd.DataFrame(standardized_data).corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")

# Add labels and title
plt.title("Correlation Coefficient Matrix Heatmap")
plt.show()

# Q4 d
from sklearn.decomposition import PCA

pca = PCA(svd_solver="full", n_components=0.95, random_state=5764)


pca_data = pca.fit_transform(standardized_data)

exp_var_pca = pca.explained_variance_ratio_

print(f'Explained variance ratio:', exp_var_pca.round(3))

# Q4 e
cum_sum_eigenvalues = np.cumsum(exp_var_pca) * 100

index_95_variance = np.argmax(cum_sum_eigenvalues >= 95)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cum_sum_eigenvalues) + 1), cum_sum_eigenvalues, marker='o', linestyle='-', color='blue')
plt.axhline(y=95, color='black', linestyle='--', label='95% Explained Variance')
plt.axvline(x=index_95_variance + 1, color='red', linestyle='--', label=f'Optimal Features: {index_95_variance + 1}')

plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.legend()
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print the first five rows of the reduced features
pca_data_df = pd.DataFrame(pca_data, columns=[f'PC_{i}' for i in range(1, pca_data.shape[1] + 1)])
print('The first five rows of the reduced features:\n', pca_data_df.head())

condition_number_before = np.linalg.cond(numerical_data)
print(f'The condition number before standardization: {condition_number_before:.2f}')

condition_number_after = np.linalg.cond(standardized_data)
print(f'The condition number after standardization: {condition_number_after:.2f}')

print(f'The difference: {(condition_number_before - condition_number_after):.2f}')

import numpy as np
from scipy.stats import kstest


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    return stats, p


# Perform K-S test for 'Price'
ks_stat_price, p_value_price = ks_test(sub_df['Price'], 'Raw')

# Interpret the K-S test result for 'Price'
if p_value_price < 0.01:
    result_price = "not normal"
else:
    result_price = "normal"

# Display K-S test results for 'Price'
print("K-S test for Price: statistics = {:.2f}, p-value = {:.2f}".format(ks_stat_price, p_value_price))
print("K-S test: Price dataset looks {}".format(result_price))

import matplotlib.pyplot as plt

# Q16 - Create a boxplot of 'Price'
plt.figure(figsize=(8, 6))
plt.boxplot(clean_df['Price'], vert=False)

# Add labels and title
plt.xlabel('Price')
plt.title('Boxplot of Price with Outliers')

# Display the plot
plt.show()

# Q17 - Calculate Q1, Q3, and IQR for 'Price'
Q1_price = clean_df['Price'].quantile(0.25)
Q3_price = clean_df['Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price

# Identify outliers for 'Price'
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

# Remove outliers from the dataset
cleaned_df_price = clean_df[(clean_df['Price'] >= lower_bound_price) & (clean_df['Price'] <= upper_bound_price)]

# Q17 - Create a boxplot of cleaned 'Price'
plt.figure(figsize=(8, 6))
plt.boxplot(cleaned_df_price['Price'], vert=False)

# Add labels and title
plt.xlabel('Price')
plt.title('Boxplot of Cleaned Price')

# Display the plot
plt.show()

from scipy.stats import shapiro


def shapiro_test(x):
    stats, p = shapiro(x)
    return stats, p


# Perform Shapiro-Wilk test for 'Price'
stat_price, p_value_price = shapiro_test(cleaned_df_price['Price'])

# Interpret the Shapiro-Wilk test result for 'Price'
if p_value_price < 0.01:
    result_price = "not normal"
else:
    result_price = "normal"

# Display Shapiro-Wilk test results for 'Price'
print("Shapiro-Wilk test for Price: statistics = {:.2f}, p-value = {:.2f}".format(stat_price, p_value_price))
print("Shapiro-Wilk test: Price data looks {}".format(result_price))

from scipy.stats import shapiro, kstest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize the 'Price' column
scaler = StandardScaler()
standard_Price = scaler.fit_transform(sub_df[['Price']])
print(standard_Price)


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = shapiro(x.flatten())  # Change this line
    return stats, p


# Perform Shapiro-Wilk test for standardized 'Price'
stat_price_std, p_value_price_std = ks_test(standard_Price, 'Raw')

# Interpret the Shapiro-Wilk test result for standardized 'Price'
if p_value_price_std < 0.01:
    result_price_std = "not normal"
else:
    result_price_std = "normal"

# Display Shapiro-Wilk test results for standardized 'Price'
print("Shapiro-Wilk test for Standardized Price: statistics = {:.2f}, p-value = {:.2f}".format(stat_price_std,
                                                                                               p_value_price_std))
print("Shapiro-Wilk test: Standardized Price data looks {}".format(result_price_std))

# In[56]:


from scipy.stats import normaltest


def da_k_squared_test(x):
    stats, p = normaltest(x)
    return stats, p


# Perform Shapiro-Wilk test for standardized 'Price'
da_k_squared_stat_price, p_value_price_std = da_k_squared_test(clean_df['Price'])

# Interpret the Shapiro-Wilk test result for standardized 'Price'
if p_value_price_std < 0.01:
    result_price_std = "not normal"
else:
    result_price_std = "normal"

# Display Shapiro-Wilk test results for standardized 'Price'
print("D'Agostino's K^2 test for Price: statistics = {:.2f}, p-value = {:.2f}".format(da_k_squared_stat_price,
                                                                                      p_value_price_std))
print("D'Agostino's K^2 test: Price looks {}".format(result_price_std))

# In[ ]:




