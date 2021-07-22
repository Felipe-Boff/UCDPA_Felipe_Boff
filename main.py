# 1 Real world scenario = ok +1
# importing packages
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from collections import Counter
import seaborn as sns
from pandas_profiling import ProfileReport


# setting up the display of columns/rows/width to better visualize the data frame
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',42)
pd.set_option('display.width',10000)

# 2 Importing data = ok +1
# importing the data frame and displaying the head/info/describe
df = pd.read_csv('/Users/felipedemoraesboff/Documents/UCD_final_project/tripadvisor_european_restaurants.csv',low_memory=False)
df = pd.DataFrame(df)
print(df.head(10))
print(df.info())
print(df.describe())

# 3 Analysing data
# 3.1 Regex = ok +1
#meter uma funcao pra def key_word_split(df,col)
data = list(df["meals"].fillna('Not listed').str.strip().str.split(","))
flattened  = [val for sublist in data for val in sublist]
flattened_strip = []
for word in flattened:
        flattened_strip.extend(word.strip().split(','))
meal_keyword_df  = pd.Series(flattened_strip)
print(meal_keyword_df.value_counts())



df['meals'] = df['meals'].str.replace(", ",",")
df['cuisines'] = df['cuisines'].str.replace(", ",",")
df['top_tags'] = df['top_tags'].str.replace(", ",",")
df['awards'] = df['awards'].str.replace(", ",",")
#print(df['meals'])
#print(df.head(10))

print(df[df['restaurant_name']=="McDonald's"].head(10))


# 3.2 replace missing or dropping duplicates = ok +1
# Data cleaning - taking a look into the missing values and dropping them as needed
NaN = df.isna().sum()
print(NaN)
# dropping columns that are not of interest ('popularity_generic', 'original_open_hours') or have too many NaNs like ('keywords','atmosphere')
# popularity_generic is very similar to popularity_detailed in some cases is identical
df = df.drop(['popularity_generic','original_open_hours','keywords','atmosphere', 'original_location','claimed','address','default_language'], axis = 1)
# dropping rows with missing values
df = df.dropna(subset=['cuisines','food','city','latitude','longitude','price_level',
                       'popularity_detailed','top_tags','avg_rating','service','value','open_days_per_week'], axis = 0)
df = df.reset_index(drop=True)
# Filling in the missing data from Region and Province columns
df['region'].fillna('Other Region', inplace=True)
df['province'].fillna('Other Province', inplace=True)
#print(df['region'].value_counts()[:20])
#print(df['province'].value_counts()[:20])
# checking NaN after dropping the missing values
NaN_updated = df.isna().sum()
print(NaN_updated)
print(df.shape)
obj_cols = df.dtypes[(df.dtypes == 'object')].index
print(df[obj_cols].describe())
#print(df['country'].value_counts(normalize=True))

# 3.3 Iterators

# 3.4 Merge dataframe
# Creating a column for the average price using the price range column as a reference
df['min_range'] = df['price_range'].str.split('-').str[0].str.replace('€', '').str.replace(',', '')
df['min_range'] = pd.to_numeric(df['min_range'], errors='coerce')
df['max_range'] = df['price_range'].str.split('-').str[1].str.replace('€', '').str.replace(',', '')
df['max_range'] = pd.to_numeric(df['max_range'], errors='coerce')
df['avg_price'] = (df['min_range'] + df['max_range']) / 2
# drop the fields used for average_price calculation
df.drop(['min_range', 'max_range'], axis=1, inplace=True)
print(df.head(5))

#grouping the data by country
countries_df = df.groupby('country').agg(
    restaurants_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size),
    open_days_per_week=pd.NamedAgg(column='open_days_per_week', aggfunc=np.mean),
    open_hours_per_week=pd.NamedAgg(column='open_hours_per_week', aggfunc=np.mean),
    working_shifts_per_week=pd.NamedAgg(column='working_shifts_per_week', aggfunc=np.mean),
    avg_rating=pd.NamedAgg(column='avg_rating', aggfunc=np.mean),
    reviews_count=pd.NamedAgg(column='total_reviews_count', aggfunc=np.sum),
    median_price=pd.NamedAgg(column='avg_price', aggfunc=np.median))
countries_df.reset_index(level=0, inplace=True)

countries_df = countries_df[['country', 'restaurants_count', 'open_days_per_week', 'open_hours_per_week',
                             'working_shifts_per_week', 'avg_rating', 'reviews_count', 'median_price']]
countries_df['reviews_per_restaurant'] = countries_df['reviews_count'] / countries_df['restaurants_count']

#merging dfs
# restaurants vegetarian_friendly
vegetarian_df = df[df['vegetarian_friendly'] == 'Y'].groupby('country').agg(
    vegetarian_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size)).reset_index(level=0)
countries_df = pd.merge(countries_df, vegetarian_df, how='inner',left_on='country', right_on='country')
countries_df['vegetarian_count_perc'] = countries_df['vegetarian_count'] / countries_df['restaurants_count']

# vegan_options
vegan_df = df[df['vegan_options'] == 'Y'].groupby('country').agg(
    vegan_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size)).reset_index(level=0)
countries_df = pd.merge(countries_df, vegan_df, how='inner',left_on='country', right_on='country')
countries_df['vegan_count_perc'] = countries_df['vegan_count'] / countries_df['restaurants_count']

# gluten_free
gluten_free_df = df[df['gluten_free'] == 'Y'].groupby('country').agg(
    gluten_free_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size)).reset_index(level=0)
countries_df = pd.merge(countries_df, gluten_free_df, how='inner',left_on='country', right_on='country')
countries_df['gluten_free_count_perc'] = countries_df['gluten_free_count'] / countries_df['restaurants_count']

# dropping the count fields that have been used to calculate the percentages
countries_df.drop(['vegetarian_count', 'vegan_count', 'gluten_free_count'], axis=1, inplace=True)



print(countries_df.head(10))

profile = ProfileReport(countries_df)
print(profile)
# 4 Python
# 4.1 Functions - create reusable code


# 4.2 Use Functions from Numpy or Scipy

# 4.3 Use a Dic or a List = ok +1
# see line 26



# 5 Machine Learning
# 5.1 Supervised Learning
# 5.2 Hyper parameter Tuning or Boosting
# 5.3 Analysing


# 6 Visualize =

#sns.relplot(x = countries_df['open_hours_per_week'], y = countries_df['avg_rating'], data = countries_df,
          #  hue = countries_df['restaurants_count'])
#plt.show()
x = countries_df['open_hours_per_week']
y = countries_df['avg_rating']
points_size = countries_df['restaurants_count']/100
colors = countries_df['avg_rating']

plt.scatter(x, y, s=points_size, c=colors)
plt.title("Scatter Plot with increase in size of scatter points ", fontsize=12)
plt.xlabel('x-axis', fontsize=12)
plt.ylabel('y-axis', fontsize=12)
plt.show()



fig_1 = df['country'].value_counts().sort_values().plot(kind='barh',figsize=(7, 6))
plt.xlabel("Country", labelpad=14)
plt.ylabel("Restaurant Count", labelpad=14)
plt.title("Breakdown of the number of restaurants in each Country ", y=1.02);
plt.show()

fig_2 = df['avg_rating'].plot(kind='hist')
plt.xlabel("avg rating", labelpad=14)
plt.title("Restaurant average rating")
plt.show()

# 7 Create Insights







#data_1 = df[df['country'] == 'Italy']
#y = df['avg_rating']
#x = df['open_hours_per_week']
#hue
#print(y)
#print(x)
#sns.scatterplot(x, y , data = data_1, hue=)
#plt.show()

#x = np.random.rand(N)
#y = np.random.rand(N)
#colors = np.random.rand(N)
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#plt.show()

#meal_list_final = set(flattened)
#print(meal_list_final)



