# importing packages
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# setting up the display of columns/rows/width to better visualize the data frame
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 42)
pd.set_option('display.width', 1000)

# 2 Importing data = ok +1
# importing the data frame and displaying the head/info/describe
df = pd.read_csv('/Users/felipedemoraesboff/Documents/UCD_final_project/tripadvisor_european_restaurants.csv',
                 low_memory=False)
df = pd.DataFrame(df)
print(df.head(10))
print(df.info())
print(df.describe())
print(df.columns)

# Checking Number of restaurants in a given Country
# px plot
country = df.groupby('country')['restaurant_link'].count().reset_index().rename(
    columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(23)
fig_1 = px.bar(country, x='#_Restaurants', y='country', color='country', height=700,
               title='Total number of restaurants based on Country', orientation='h', text='#_Restaurants')
fig_1.show()

# # Seaborn plot to show the same as above
sns.set_color_codes('pastel')
fig_2 = sns.barplot(x='#_Restaurants', y='country', data=country, palette="deep")
## commenting out the plt.show as it's breaking the code here
# plt.show()

# # Checking Number of restaurants in the top 20 Cities
country = df.groupby('city')['restaurant_link'].count().reset_index().rename(
    columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(20)
# Seaborn plot
fig_3 = px.bar(country, x='#_Restaurants', y='city', color='city', height=700,
               title='Cities in Europe with biggest number of restaurants', orientation='h')
fig_3.show()

# Seaborn barplot to show the same graph as above
sns.set_color_codes('pastel')
fig_4 = sns.barplot(x='#_Restaurants', y='city', data=country, palette="deep")
## commenting out the plt.show as it's breaking the code here
# plt.show()


# 3 Analysing data
# 3.1 Regex = ok +1
# it has always fascinated me restaurant naming conventions where some will
# have only A-Z and others will have A-Z and 0-9
regex_n = '.*[0-9]+.*'
restaurant_w_num = df['restaurant_name']
count_n = 0
for restaurant in restaurant_w_num:
    if re.fullmatch(regex_n, restaurant):
        count_n += 1
print(count_n)

regex_a_z = '^[a-zA-Z_ ]*$'
count_a_z = 0
for restaurant in restaurant_w_num:
    if re.fullmatch(regex_a_z, restaurant):
        count_a_z += 1
print(count_a_z)

df_restaurants_0_9 = df[df['restaurant_name'].str.count(regex_n) > 0]
df_restaurants_a_z = df[df['restaurant_name'].str.count(regex_a_z) > 0]
print(df_restaurants_0_9.head(5))
print(df_restaurants_a_z.head(5))

# 3.2 replace missing / dropping duplicates / creating new columns = ok +1

# Creating a column for the average price using the price range column as a reference
df['min_range'] = df['price_range'].str.split('-').str[0].str.replace('€', '').str.replace(',', '')
df['min_range'] = pd.to_numeric(df['min_range'], errors='coerce')
df['max_range'] = df['price_range'].str.split('-').str[1].str.replace('€', '').str.replace(',', '')
df['max_range'] = pd.to_numeric(df['max_range'], errors='coerce')
df['avg_price'] = (df['min_range'] + df['max_range']) / 2
print('Initially there were {} NaN values for the avg_price column'.format(df['avg_price'].isna().sum()))
# df['avg_price'] = df['avg_price'].fillna(df.groupby('city')['avg_price'].transform('median'))
df['avg_price'] = df['avg_price'].fillna(df.groupby('city')['avg_price'].transform('median'))
print('There are still {} NaN values after filling the missing avg_price grouped by city'.format(
    df['avg_price'].isna().sum()))
df['avg_price'] = df['avg_price'].fillna(df.groupby('region')['avg_price'].transform('median'))
print('There are still {} NaN values after filling the missing avg_price grouped by region'.format(
    df['avg_price'].isna().sum()))
df['avg_price'] = df['avg_price'].fillna(df.groupby('province')['avg_price'].transform('median'))
print('There are still {} NaN values after filling the missing avg_price grouped by province'.format(
    df['avg_price'].isna().sum()))
df['avg_price'] = df['avg_price'].fillna(df.groupby('country')['avg_price'].transform('median'))
print('There are still {} NaN values after filling the missing avg_price grouped by country'.format(
    df['avg_price'].isna().sum()))
# drop the fields used for average_price calculation
df.drop(['min_range', 'max_range'], axis=1, inplace=True)
print(df.head(5))
print(df['country'].nunique())

# Data cleaning - taking a look into the missing values and dropping them as needed
NaN = df.isna().sum()
print(NaN)
# Filling in the missing data from Region and Province columns
df['region'].fillna('Other Region', inplace=True)
df['province'].fillna('Other Province', inplace=True)

# dropping columns that are not of interest ('popularity_generic', 'original_open_hours') or have too many NaNs like ('keywords','atmosphere')
# popularity_generic is very similar to popularity_detailed in some cases is identical
df = df.drop(['popularity_generic', 'original_open_hours', 'keywords', 'atmosphere',
              'original_location', 'claimed', 'address', 'default_language'], axis=1)
# dropping rows with missing values
df = df.dropna(subset=['cuisines', 'food', 'city', 'latitude', 'longitude', 'price_level',
                       'popularity_detailed', 'top_tags', 'avg_rating', 'service', 'value', 'open_days_per_week'],
               axis=0)
df = df.reset_index(drop=True)
# checking NaN after dropping the missing values
NaN_updated = df.isna().sum()
print(NaN_updated)
print(df.shape)
obj_cols = df.dtypes[(df.dtypes == 'object')].index
print(df[obj_cols].describe())
print(df['country'].value_counts(normalize=True))
print(df.head(10))
print(df.info())
print(df.describe())
print(df.columns)

# px plot after NaNs fix and dropped columns
country = df.groupby('country')['restaurant_link'].count().reset_index().rename(
    columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(23)
fig_1 = px.bar(country, x='#_Restaurants', y='country', color='country', height=700,
               title='Total number of restaurants based on Country - clean database', orientation='h',
               text='#_Restaurants')
fig_1.show()

# Seaborn plot
sns.set_color_codes('pastel')
fig_2 = sns.barplot(x='#_Restaurants', y='country', data=country, palette="deep")
## plt.show() is blocking the code from running on pycharm hence commented out
# plt.show()

# # Checking Number of restaurants in the top 20 Cities clean database
country = df.groupby('city')['restaurant_link'].count().reset_index().rename(
    columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(20)
# Seaborn plot
fig_3 = px.bar(country, x='#_Restaurants', y='city', color='city', height=700,
               title='Cities in Europe with biggest number of restaurants', orientation='h', text='#_Restaurants')
fig_3.show()

sns.set_color_codes('pastel')
fig_4 = sns.barplot(x='#_Restaurants', y='city', data=country, palette="deep")
## plt.show() is blocking the code from running on pycharm hence commented out
# plt.show()


# 3.3 Iterators
# fixing the issue of extra space after comma on some of the columns of the data frame
list_1 = ['meals', 'cuisines', 'top_tags', 'awards', 'features']
for col in list_1:
    df[col] = df[col].str.replace(", ", ",")
print(df[['meals', 'cuisines', 'top_tags', 'awards', 'features']].head(5))
# map and replace Y/N for 1/0
list_2 = ['vegetarian_friendly', 'vegan_options', 'gluten_free']
for col in list_2:
    df[col] = df[col].map({'Y': 1, 'N': 0})
print(df[['vegetarian_friendly', 'vegan_options', 'gluten_free']].head(5))


##3.4 Merge dataframe and unsing function to leverage replicable code
def aggregate_by(dataframe, col):
    aggregating_df = dataframe.groupby(col).agg(
        restaurants_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size),
        open_days_per_week=pd.NamedAgg(column='open_days_per_week', aggfunc=np.mean),
        open_hours_per_week=pd.NamedAgg(column='open_hours_per_week', aggfunc=np.mean),
        working_shifts_per_week=pd.NamedAgg(column='working_shifts_per_week', aggfunc=np.mean),
        avg_rating=pd.NamedAgg(column='avg_rating', aggfunc=np.mean),
        reviews_count=pd.NamedAgg(column='total_reviews_count', aggfunc=np.sum),
        median_price=pd.NamedAgg(column='avg_price', aggfunc=np.median))
    aggregating_df.reset_index(level=0, inplace=True)
    return aggregating_df


# grouping the data  by city
city_df = aggregate_by(df, 'city')
city_df = city_df[['city', 'restaurants_count', 'open_days_per_week', 'open_hours_per_week',
                   'working_shifts_per_week', 'avg_rating', 'reviews_count', 'median_price']]
city_df['reviews_per_restaurant'] = city_df['reviews_count'] / city_df['restaurants_count']
print(city_df.head(5))
print(city_df.info())
print(city_df.describe())

# grouping the data by regions
region_df = aggregate_by(df, 'region')
region_df = region_df[['region', 'restaurants_count', 'open_days_per_week', 'open_hours_per_week',
                       'working_shifts_per_week', 'avg_rating', 'reviews_count', 'median_price']]
region_df['reviews_per_restaurant'] = region_df['reviews_count'] / region_df['restaurants_count']
print(region_df.head(5))
print(region_df.info())
print(region_df.describe())

# grouping the data by country
countries_df = aggregate_by(df, 'country')
countries_df = countries_df[['country', 'restaurants_count', 'open_days_per_week', 'open_hours_per_week',
                             'working_shifts_per_week', 'avg_rating', 'reviews_count', 'median_price']]
countries_df['reviews_per_restaurant'] = countries_df['reviews_count'] / countries_df['restaurants_count']
print(countries_df.head(5))
print(countries_df.info())
print(countries_df.describe())


# Merging dfs and using functions
def dietary_group(col_diet, agg_col, count_name):
    return df[df[col_diet] == 1].groupby(agg_col).agg(
        **{count_name: pd.NamedAgg(column='restaurant_link', aggfunc=np.size)}).reset_index(level=0)


# restaurants vegetarian_friendly country
vegetarian_country_df = dietary_group('vegetarian_friendly', 'country', 'vegetarian_count')
countries_df = pd.merge(countries_df, vegetarian_country_df, how='inner', left_on='country', right_on='country')
countries_df['vegetarian_count_perc'] = countries_df['vegetarian_count'] / countries_df['restaurants_count']
# restaurants vegetarian_friendly city
vegetarian_city_df = dietary_group('vegetarian_friendly', 'city', 'vegetarian_count')
city_df = pd.merge(city_df, vegetarian_city_df, how='inner', left_on='city', right_on='city')
city_df['vegetarian_count_perc'] = city_df['vegetarian_count'] / city_df['restaurants_count']

# vegan_options country
vegan_country_df = dietary_group('vegan_options', 'country', 'vegan_count')
countries_df = pd.merge(countries_df, vegan_country_df, how='inner', left_on='country', right_on='country')
countries_df['vegan_count_perc'] = countries_df['vegan_count'] / countries_df['restaurants_count']
# vegan_options city
vegan_city_df = dietary_group('vegan_options', 'city', 'vegan_count')
city_df = pd.merge(city_df, vegan_city_df, how='inner', left_on='city', right_on='city')
city_df['vegan_count_perc'] = city_df['vegan_count'] / city_df['restaurants_count']

# gluten_free country
gluten_free_country_df = dietary_group('gluten_free', 'country', 'gluten_free_count')
countries_df = pd.merge(countries_df, gluten_free_country_df, how='inner', left_on='country', right_on='country')
countries_df['gluten_free_count_perc'] = countries_df['gluten_free_count'] / countries_df['restaurants_count']
# gluten_free city
gluten_free_city_df = dietary_group('gluten_free', 'city', 'gluten_free_count')
city_df = pd.merge(city_df, gluten_free_city_df, how='inner', left_on='city', right_on='city')
city_df['gluten_free_count_perc'] = city_df['gluten_free_count'] / city_df['restaurants_count']

# dropping the count fields that have been used to calculate the percentages
countries_df.drop(['vegetarian_count', 'vegan_count', 'gluten_free_count'], axis=1, inplace=True)
print(countries_df.head(10))

# Top 20 European cities dietary preferences avg_rating vs. open hours
top20_city_df = city_df.sort_values('restaurants_count', ascending=False).head(20)
print(top20_city_df)
fig_5 = go.Figure(data=go.Scatter(x=top20_city_df['open_hours_per_week'], y=countries_df['avg_rating'],
                                  marker=dict(size=top20_city_df['restaurants_count'] / 50,
                                              color=top20_city_df['avg_rating']),
                                  mode='markers+text',
                                  text=top20_city_df['city'], textposition='top center', textfont=dict(size=9),
                                  hoverlabel=dict(namelength=0),
                                  # removes the trace number off to the side of the tooltip box
                                  hovertemplate='%{text}:<br>%{x:.2f} days<br>%{y:.1f} hours'))
fig_5.update_layout(title='Average rating based on open hours per week (size by restaurants count)',
                    title_x=0.5, legend=dict(yanchor='bottom', y=-0.15, xanchor='left', x=0,
                                             font=dict(size=10), orientation='h'))
fig_5['layout']['xaxis']['title'] = 'Open hours per week'
fig_5['layout']['yaxis']['title'] = 'Average rating'
fig_5.show()
# Top20 European cities reviews based on dietary preferences
fig = plotly.subplots.make_subplots(rows=1, cols=3, subplot_titles=('Vegetarian', 'Vegan', 'Gluten-free'),
                                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]])
fig.add_trace(go.Scatter(x=top20_city_df['vegetarian_count_perc'], y=top20_city_df['reviews_per_restaurant'],
                         marker=dict(size=top20_city_df['restaurants_count'] / 100,
                                     color=top20_city_df['reviews_per_restaurant']), mode='markers+text',
                         showlegend=False,
                         text=top20_city_df['city'], textposition='top center', textfont=dict(size=9),
                         hoverlabel=dict(namelength=0),  # removes the trace number off to the side of the tooltip box
                         hovertemplate='%{text}:<br>%{x:.2f} days<br>%{y:.1f} hours'), row=1, col=1)
fig.add_trace(go.Scatter(x=top20_city_df['vegan_count_perc'], y=top20_city_df['reviews_per_restaurant'],
                         marker=dict(size=top20_city_df['restaurants_count'] / 100,
                                     color=top20_city_df['reviews_per_restaurant']), mode='markers+text',
                         showlegend=False,
                         text=top20_city_df['city'], textposition='top center', textfont=dict(size=9),
                         hoverlabel=dict(namelength=0),  # removes the trace number off to the side of the tooltip box
                         hovertemplate='%{text}:<br>%{x:.2f} days<br>%{y:.1f} hours'), row=1, col=2)
fig.add_trace(go.Scatter(x=top20_city_df['gluten_free_count_perc'], y=top20_city_df['reviews_per_restaurant'],
                         marker=dict(size=top20_city_df['restaurants_count'] / 100,
                                     color=top20_city_df['reviews_per_restaurant']), mode='markers+text',
                         showlegend=False,
                         text=top20_city_df['city'], textposition='top center', textfont=dict(size=9),
                         hoverlabel=dict(namelength=0),  # removes the trace number off to the side of the tooltip box
                         hovertemplate='%{text}:<br>%{x:.2f} days<br>%{y:.1f} hours'), row=1, col=3)
fig.update_layout(title='Reviews per Restaurant based on Vegetarian %, Vegan %, and Gluten-free %', title_x=0.5,
                  legend=dict(yanchor='bottom', y=-0.15, xanchor='left', x=0,
                              font=dict(size=8), orientation='h'))
fig['layout']['xaxis']['title'] = 'Vegetarian %'
fig['layout']['xaxis2']['title'] = 'Vegan %'
fig['layout']['xaxis3']['title'] = 'Gluten-free %'
fig.show()

# histogram of the median price on the top20 european cities
sns.set(style="darkgrid")
sns.histplot(data=top20_city_df, x="median_price", kde=True)


# this plt.show() is blowing the code from running on my pycharm hence it's commmented out
# plt.show()


# 4.2 Use Functions from Numpy or Scipy

# 4.3 Use a Dic or a List = ok +1
# see line 26


############### 5 MACHINE LEARNING ###############
# 5.1 Supervised Learning
# 5.2 Hyper parameter Tuning or Boosting
# imputing the missing values

################ CLUSTERING KMEANS ###############
# Use a Dictionary or Lists to store Data
# Want to show the total number of movies by each Genre

def split(data, col):
    new_data = data[col].str.get_dummies(',')
    return new_data


cuisines_unique = split(df, 'cuisines')
############### Merging new columns with the dataframe ###############
cuisines_types = pd.concat([df, cuisines_unique], axis=1)
cuisines_types = cuisines_types[['city', 'open_days_per_week', 'open_hours_per_week',
                                 'working_shifts_per_week', 'food', 'service', 'value',
                                 'avg_rating', 'avg_price', 'European', 'French',
                                 'Mediterranean', 'British', 'Italian', 'Bar', 'Cafe',
                                 'Pub', 'Asian', 'Pizza', 'Seafood', 'Fast food', 'Spanish',
                                 'International', 'German', 'Greek', 'Indian', 'Healthy',
                                 'Central European', 'Chinese']]
# cuisines_types = cuisines_types[cuisines_types['city'] == 'Paris'].reset_index()


cuisines_types = cuisines_types[cuisines_types['city'].isin(['Paris', 'Rome', 'Madrid', 'Milan', 'Prague', 'Amsterdam',
                                                             'Lisbon', 'Vienna', 'Munich', 'Budapest', 'Lyon', 'Dublin',
                                                             'Manchester', 'Stockholm',
                                                             'Birmingham', 'Copenhagen', 'Marseille', 'Porto', 'Athens',
                                                             'Nice'])].reset_index()
print(cuisines_types.head(10))

# Top 20 cusine types in European restaurants pie chart - done
print(cuisines_unique.sum().sort_values(ascending=False).head(20))
plt.figure(figsize=(20, 10))
names = ['European', 'French', 'Mediterranean', 'British', 'Italian', 'Bar', 'Cafe', 'Pub', 'Asian', 'Pizza',
         'Seafood', 'Fast food', 'Spanish', 'International', 'German', 'Greek', 'Indian', 'Healthy', 'Central European',
         'Chinese']
plt.pie(cuisines_unique.sum().sort_values(ascending=False).head(20), labels=names, autopct='%1.f%%')
plt.ylabel('Cuisines')
plt.xlabel('Cuisines %')
plt.title('Top 20 Cuisines in European restaurants')
## plt.show is blocking the code here as well, therefore I'm commenting it out
# plt.show()


# Selecting features to cluster
# first test using Kmeans
# features = df_knn[['avg_rating','median_price','open_hours_per_week','restaurants_count']].astype(int)

# df for city = Paris
# df_knn = df_knn_all[df_knn_all['city']=='Paris']

# df_knn_city = aggregate_by(df_knn_all, 'city')
# df_knn_city = df_knn_city.drop([], axis=1)

# these features are working on the Kmeans for city = Paris
# features = cuisines_types[['European','French','Mediterranean','British','Italian','Bar','Cafe',
#                            'Pub','Asian','Pizza', 'Seafood','Fast food','Spanish','International',
#                            'German','Greek','Indian','Healthy','Central European','Chinese']].astype(int)


features = cuisines_types[['open_days_per_week', 'open_hours_per_week', 'working_shifts_per_week',
                           'food', 'service', 'value', 'avg_rating', 'avg_price', 'European',
                           'French', 'Mediterranean', 'British', 'Italian', 'Bar', 'Cafe',
                           'Pub', 'Asian', 'Pizza', 'Seafood', 'Fast food', 'Spanish',
                           'International', 'German', 'Greek', 'Indian', 'Healthy',
                           'Central European', 'Chinese']].astype(int)

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# # # Using TSNE
tsne = TSNE(n_components=2)
rating_transform = tsne.fit_transform(scaled_data)

################ KMeans -  The Elbow Method ###############
wcss = []
K = range(1, 50)
for k in K:
    kmean = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmean.fit(scaled_data)
    wcss.append(kmean.inertia_)
fig = px.line(x=K, y=wcss, title='Optimal K from The Elbow Method',
              labels={'x': 'Number of Clusters', 'y': 'Distortions'})
fig.show()

############## KMEANS ###############
cluster = KMeans(n_clusters=19, init='k-means++', random_state=42)
y_kmeans = cluster.fit_predict(scaled_data)

df_tsne = pd.DataFrame(np.column_stack((rating_transform, y_kmeans,
                                        cuisines_types['avg_price'], cuisines_types['city'])),
                       columns=['X', 'Y', 'Group', 'avg_price', 'city'])

fig = px.scatter(df_tsne, x='X', y='Y', hover_data=['avg_price', 'city'], color=y_kmeans,
                 color_discrete_sequence=px.colors.cyclical.IceFire)
fig.show()

# # Boosting
print(df.isna().sum())
df_boosting = df[
    ['vegetarian_friendly', 'open_days_per_week', 'open_hours_per_week', 'working_shifts_per_week', 'avg_rating',
     'total_reviews_count', 'food',
     'service', 'value', 'avg_price']]
print(df_boosting.head(10))
df_boosting_cols = ['avg_rating', 'total_reviews_count', 'food', 'service', 'value', 'avg_price']
x = df_boosting.drop(labels='vegetarian_friendly', axis=1)
y = df['vegetarian_friendly']
scaler = StandardScaler()
scaled_df = scaler.fit_transform(x)
train_x, test_x, train_y, test_y = train_test_split(scaled_df, y, test_size=0.3, random_state=42)
model = XGBClassifier(objective='binary:logistic')
model.fit(train_x, train_y)
# cheking training accuracy
y_pred = model.predict(train_x)
predictions = [round(value) for value in y_pred]
accuracy_train = accuracy_score(train_y, predictions)
print('XGBC accuracy train {}'.format(accuracy_train))
# cheking initial test accuracy
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
accuracy_test = accuracy_score(test_y, predictions)
print('XGBC accuracy test {}'.format(accuracy_test))
print(test_x[0])
# commented the following best params as it was my first run
# param_grid = {
#    'learning_rate': [1, 0.1, 0.01, 0.001],
#    'max_depth': [3, 5, 10, 20],
#    'n_estimators': [10, 50, 100, 200]
# }
# # second attempt on the param_grids
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [5, 10, 20],
    'n_estimators': [50, 100, 200]
}
grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)
grid.fit(train_x, train_y)
print(grid.best_params_)
##  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
# Create new model using the same parameters
new_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)
new_model.fit(train_x, train_y)
y_pred_new = new_model.predict(test_x)
predictions_new = [round(value) for value in y_pred_new]
accuracy_new = accuracy_score(test_y, predictions_new)
print('XGBC best parameters accuracy {}'.format(accuracy_new))
# As we have increased the accuracy of the model, we'll save this model
filename = 'xgboost_model.pickle'
pickle.dump(new_model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
# we'll save the scaler object as well for prediction
filename_scaler = 'scaler_model.pickle'
pickle.dump(scaler, open(filename_scaler, 'wb'))
scaler_model = pickle.load(open(filename_scaler, 'rb'))
# Trying a random prediction
# is this a vegetarian restaurant Y/N
d = scaler_model.transform([[6.0, 24.0, 6.0, 4.5, 133.0, 4.5, 4.5, 4.5, 22.5]])
pred = loaded_model.predict(d)
print('This data belongs to class :', pred[0])

# DecisionTreeClassifier
df_tree = df[['vegetarian_friendly', 'open_days_per_week', 'open_hours_per_week', 'working_shifts_per_week',
              'avg_rating', 'total_reviews_count', 'food', 'service', 'value', 'avg_price']]
print(df_tree.info())
X_tree = df_tree.drop(columns='vegetarian_friendly')
y_tree = df_tree['vegetarian_friendly']

x_train, x_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.30, random_state=42)
# let's first visualize the tree on the data without doing any pre processing
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(x_train, y_train)
# accuracy of our classification tree
print('Test score Decision Tree with min_samples split {}'.format(clf.score(x_test, y_test)))
# let's first visualize the tree on the data without doing any pre processing
clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=24, min_samples_leaf=1)
clf2.fit(x_train, y_train)
print('Test score Decision Tree with  criterion, max_depth and min_samples leaf {}'.format(clf2.score(x_test, y_test)))
rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(x_train, y_train)
print('Test score Random Forest Classifier {}'.format(rand_clf.score(x_test, y_test)))
# tuning three hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    "n_estimators": [20, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 10, 1),
    'min_samples_leaf': range(1, 5, 1),
    'min_samples_split': range(2, 5, 1),
    'max_features': ['auto']
}
grid_search = GridSearchCV(estimator=rand_clf, param_grid=grid_param, cv=5, n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)
# let's see the best parameters as per our grid search
print(grid_search.best_params_)
### {'criterion': 'gini', 'max_depth': 9, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
rand_clf = RandomForestClassifier(criterion='gini', max_depth=9, max_features='auto',
                                  min_samples_leaf=2,
                                  min_samples_split=2,
                                  n_estimators=100,
                                  random_state=42)

rand_clf.fit(x_train, y_train)
print('Test score for best parameters using Random forest Classifier {}'.format(rand_clf.score(x_test, y_test)))

# tuning three hyperparameters, passing the different values
grid_param = {
    "n_estimators": [90, 100, 115],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [4, 5, 6, 7, 8],
    'max_features': ['auto', 'log2']
}

grid_search = GridSearchCV(estimator=rand_clf, param_grid=grid_param, cv=5, n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)

## {'criterion': 'gini', 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}
rand_clf = RandomForestClassifier(criterion='gini', max_features='auto', min_samples_leaf=1,
                                  min_samples_split=4, n_estimators=100, random_state=42)

rand_clf.fit(x_train, y_train)
print('Test score using best parameters on the Random forest classifier {}'.format(rand_clf.score(x_test, y_test)))
