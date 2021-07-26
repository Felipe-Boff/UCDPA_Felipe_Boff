# 1 Real world scenario = ok +1
# importing packages
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from collections import Counter
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
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
# country = df.groupby('country')['restaurant_link'].count().reset_index().rename(columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(23)
# fig_1 = px.bar(country, x='#_Restaurants', y='country', color='country', height=700,
#              title='Total number of restaurants based on Country', orientation='h')
# fig_1.show()
#
# # Seaborn plot
# sns.set_color_codes('pastel')
# fig_2 = sns.barplot(x='#_Restaurants', y='country', data=country, palette="deep")
#
#
# # Checking Number of restaurants in the top 20 Cities
# country = df.groupby('city')['restaurant_link'].count().reset_index().rename(columns={'restaurant_link': '#_Restaurants'}).sort_values('#_Restaurants', ascending=False).head(20)
# # Seaborn plot
# fig_3 = px.bar(country, x='#_Restaurants', y='city', color='city', height=700,
#              title='Cities in Europe with biggest number of restaurants', orientation='h')
# fig_3.show()
#
# sns.set_color_codes('pastel')
# fig_4 = sns.barplot(x='#_Restaurants', y='city', data=country, palette="deep")
# plt.show()


# 3 Analysing data
# 3.1 Regex = ok +1
# it has always fascinated me restaurant naming conventions where some will
# have only A-Z and others will have A-Z and 0-9
# regex_n = '.*[0-9]+.*'
# restaurant_w_num = df['restaurant_name']
# count_n = 0
# for restaurant in restaurant_w_num:
#     if re.fullmatch(regex_n, restaurant):
#         count_n += 1
# print(count_n)
#
# regex_a_z = '^[a-zA-Z_ ]*$'
# count_a_z = 0
# for restaurant in restaurant_w_num:
#     if re.fullmatch(regex_a_z, restaurant):
#         count_a_z += 1
# print(count_a_z)
#
# df_restaurants_0_9 = df[df['restaurant_name'].str.count(regex_n) > 0]
# df_restaurants_a_z = df[df['restaurant_name'].str.count(regex_a_z) > 0]
# print(df_restaurants_0_9.head(5))
# print(df_restaurants_a_z.head(5))

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


##3.4 Merge dataframe
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


# Merging dfs
def dietary_group(col_diet, agg_col, count_name):
    return df[df[col_diet] == 1].groupby(agg_col).agg(
        **{count_name: pd.NamedAgg(column='restaurant_link', aggfunc=np.size)}).reset_index(level=0)


# restaurants vegetarian_friendly
vegetarian_country_df = dietary_group('vegetarian_friendly', 'country', 'vegetarian_count')
countries_df = pd.merge(countries_df, vegetarian_country_df, how='inner', left_on='country', right_on='country')
countries_df['vegetarian_count_perc'] = countries_df['vegetarian_count'] / countries_df['restaurants_count']

# vegan_options
vegan_country_df = dietary_group('vegan_options','country','vegan_count')
countries_df = pd.merge(countries_df, vegan_country_df, how='inner', left_on='country', right_on='country')
countries_df['vegan_count_perc'] = countries_df['vegan_count'] / countries_df['restaurants_count']

# gluten_free
gluten_free_country_df = dietary_group('gluten_free','country','gluten_free_count')
countries_df = pd.merge(countries_df, gluten_free_country_df, how='inner', left_on='country', right_on='country')
countries_df['gluten_free_count_perc'] = countries_df['gluten_free_count'] / countries_df['restaurants_count']

# dropping the count fields that have been used to calculate the percentages
countries_df.drop(['vegetarian_count', 'vegan_count', 'gluten_free_count'], axis=1, inplace=True)
print(countries_df.head(10))

# fig = go.Figure(data=go.Scatter(x=countries_df['open_hours_per_week'], y=countries_df['avg_rating'],
#                                 marker=dict(size=countries_df['restaurants_count']/1000,
#                                             color=countries_df['avg_rating']),
#                                 mode='markers+text',
#                                 text=countries_df['country'], textposition='top center', textfont=dict(size=9),
#                                 hoverlabel=dict(namelength=0), # removes the trace number off to the side of the tooltip box
#                                 hovertemplate='%{text}:<br>%{x:.2f} days<br>%{y:.1f} hours'))
# fig.update_layout(title='Average rating based on open hours per week (size by restaurants count)',
#                   title_x=0.5, legend=dict(yanchor='bottom', y=-0.15, xanchor='left', x=0,
#                                            font=dict(size=10), orientation='h'))
# fig['layout']['xaxis']['title'] = 'Open hours per week'
# fig['layout']['yaxis']['title'] = 'Average rating'
# fig.show()




# profile = ProfileReport(countries_df)
# print(profile)
# 4 Python
# 4.1 Functions - create reusable code


# 4.2 Use Functions from Numpy or Scipy

# 4.3 Use a Dic or a List = ok +1
# see line 26


############### 5 MACHINE LEARNING ###############
# 5.1 Supervised Learning
# 5.2 Hyper parameter Tuning or Boosting
# imputing the missing values

################ CLUSTERING KMEANS ###############
# Use a Dictionary or Lists to store Data
cuisines_list = list(df['cuisines'].str.strip().str.split(','))
flat_cuisines_list = []
for sublit in cuisines_list:
    for cuisines in sublit:
        flat_cuisines_list.append((cuisines))
#print(set(flat_cuisines_list))
cuisines_df = pd.Series(flat_cuisines_list)
print(cuisines_df.value_counts().sort_values(ascending=False))

df_knn = city_df[['city', 'restaurants_count', 'open_days_per_week', 'open_hours_per_week',
                   'working_shifts_per_week', 'avg_rating', 'reviews_count', 'median_price']]

def split(data,col):
    new_data = data[col].str.get_dummies(',')
    return new_data
cuisines_split = split(df,'cuisines')

############### Merging new columns with the dataframe ###############
#df_knn = pd.concat([df,cuisines_split], axis=1, sort=False)

# Selecting features to cluster
features = df_knn[['avg_rating','median_price','open_hours_per_week','restaurants_count']].astype(int)
# features = df_knn[['European','French','Mediterranean','British','Italian','Bar','Cafe','Pub','Asian','Pizza',
#                'Seafood','Fast food','Spanish','International','German','Greek','Indian','Healthy','Central European',
#                'Chinese','Portuguese','Grill','American','Japanese','Barbecue','Gastropub','Steakhouse','Contemporary','Dutch','Sushi']].astype(int)
# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# # Using TSNE
tsne = TSNE(n_components=2)
rating_transform = tsne.fit_transform(scaled_data)

################ KMeans -  The Elbow Method ###############
# wcss = []
# K = range(1,13)
# for k in K:
#     kmean = KMeans(n_clusters=k,init='k-means++',random_state=42)
#     kmean.fit(scaled_data)
#     wcss.append(kmean.inertia_)
# fig = px.line(x=K,y=wcss,title='Optimal K from The Elbow Method',
#               labels={'x':'Number of Clusters','y':'Distortions'})
# fig.show()

############## KMEANS ###############
cluster = KMeans(n_clusters=5)
predict_group = cluster.fit_predict(scaled_data)

df_tsne = pd.DataFrame(np.column_stack((rating_transform, predict_group,
                                        df_knn['reviews_count'], df_knn['city'])),
                       columns=['X', 'Y', 'Group', 'reviews_count', 'city'])

fig = px.scatter(df_tsne, x='X', y='Y', hover_data=['reviews_count', 'city'], color='Group',
                 color_discrete_sequence=px.colors.cyclical.IceFire)
fig.show()


# # Boosting
# print(df.isna().sum())
# df_boosting = df[
#     ['vegetarian_friendly', 'open_days_per_week', 'open_hours_per_week', 'working_shifts_per_week', 'avg_rating',
#      'total_reviews_count', 'food',
#      'service', 'value', 'avg_price']]
# print(df_boosting.head(10))
# df_boosting_cols = ['avg_rating', 'total_reviews_count', 'food', 'service', 'value', 'avg_price']
# x = df_boosting.drop(labels='vegetarian_friendly', axis=1)
# y = df['vegetarian_friendly']
# scaler = StandardScaler()
# scaled_df = scaler.fit_transform(x)
# train_x, test_x, train_y, test_y = train_test_split(scaled_df, y, test_size=0.3, random_state=42)
# model = XGBClassifier(objective='binary:logistic')
# model.fit(train_x, train_y)
# # cheking training accuracy
# y_pred = model.predict(train_x)
# predictions = [round(value) for value in y_pred]
# accuracy_train = accuracy_score(train_y, predictions)
# print(accuracy_train)
# # cheking initial test accuracy
# y_pred = model.predict(test_x)
# predictions = [round(value) for value in y_pred]
# accuracy_test = accuracy_score(test_y, predictions)
# print(accuracy_test)
# print(test_x[0])
# # param_grid = {
# #    'learning_rate': [1, 0.1, 0.01, 0.001],
# #    'max_depth': [3, 5, 10, 20],
# #    'n_estimators': [10, 50, 100, 200]
# # }
# # rodando ML segunda vez
# param_grid = {
#     'learning_rate': [0.1, 0.01, 0.001],
#     'max_depth': [5, 10, 20],
#     'n_estimators': [50, 100, 200]
# }
# grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)
# grid.fit(train_x, train_y)
# print(grid.best_params_)
# # Create new model using the same parameters
# new_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)
# new_model.fit(train_x, train_y)
# y_pred_new = new_model.predict(test_x)
# predictions_new = [round(value) for value in y_pred_new]
# accuracy_new = accuracy_score(test_y, predictions_new)
# print(accuracy_new)
# # As we have increased the accuracy of the model, we'll save this model
# filename = 'xgboost_model.pickle'
# pickle.dump(new_model, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
# # we'll save the scaler object as well for prediction
# filename_scaler = 'scaler_model.pickle'
# pickle.dump(scaler, open(filename_scaler, 'wb'))
# scaler_model = pickle.load(open(filename_scaler, 'rb'))
# # Trying a random prediction
# # is this a vegetarian restaurant Y/N
# d = scaler_model.transform([[6.0, 24.0, 6.0, 4.5, 133.0, 4.5, 4.5, 4.5, 22.5]])
# pred = loaded_model.predict(d)
# print('This data belongs to class :', pred[0])
#
# # DecisionTreeClassifier
# df_tree = df[['vegetarian_friendly', 'open_days_per_week', 'open_hours_per_week', 'working_shifts_per_week',
#               'avg_rating', 'total_reviews_count', 'food', 'service', 'value', 'avg_price']]
# print(df_tree.info())
# X_tree = df_tree.drop(columns='vegetarian_friendly')
# y_tree = df_tree['vegetarian_friendly']
#
# x_train, x_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.30, random_state=42)
# # let's first visualize the tree on the data without doing any pre processing
# clf = DecisionTreeClassifier(min_samples_split=2)
# clf.fit(x_train, y_train)
# # accuracy of our classification tree
# print(clf.score(x_test, y_test))
# # let's first visualize the tree on the data without doing any pre processing
# clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=24, min_samples_leaf=1)
# clf2.fit(x_train, y_train)
# print(clf2.score(x_test, y_test))
# rand_clf = RandomForestClassifier(random_state=6)
# rand_clf.fit(x_train, y_train)
# rand_clf.score(x_test, y_test)
# we are tuning three hyperparameters right now, we are passing the different values for both parameters
# grid_param = {
#     "n_estimators" : [20,50,100],
#     'criterion': ['gini', 'entropy'],
#     'max_depth' : range(2,10,1),
#     'min_samples_leaf' : range(1,5,1),
#     'min_samples_split': range(2,5,1),
#     'max_features' : ['auto']
# }
# grid_search = GridSearchCV(estimator=rand_clf,param_grid=grid_param,cv=5,n_jobs =-1,verbose = 3)
# grid_search.fit(x_train,y_train)
# let's see the best parameters as per our grid search
# print(grid_search.best_params_)
#
# rand_clf = RandomForestClassifier(criterion='gini', max_depth=9, max_features='auto',
#                                   min_samples_leaf=3,
#                                   min_samples_split=2,
#                                   n_estimators=50,
#                                   random_state=42)
#
# rand_clf.fit(x_train, y_train)
# print(rand_clf.score(x_test, y_test))

# we are tuning three hyperparameters right now, we are passing the different values for both parameters
# grid_param = {
#     "n_estimators" : [90,100,115],
#     'criterion': ['gini', 'entropy'],
#     'min_samples_leaf' : [1,2,3,4,5],
#     'min_samples_split': [4,5,6,7,8],
#     'max_features' : ['auto','log2']
# }
# # best_params: {'criterion': 'gini', 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 90}
# grid_search = GridSearchCV(estimator=rand_clf,param_grid=grid_param,cv=5,n_jobs =-1,verbose = 3)
# grid_search.fit(x_train,y_train)
# print(grid_search.best_params_)

# rand_clf = RandomForestClassifier(criterion='gini', max_features='auto', min_samples_leaf=2,
#                                   min_samples_split=4, n_estimators=90, random_state=42)
#
# rand_clf.fit(x_train, y_train)
# print(rand_clf.score(x_test, y_test))

# print(df_boosting.columns)
# print(df_boosting.isna().sum())
# print(df['avg_price'].median())

# df_boosting['meals'] = df_boosting['meals'].fillna(df_boosting['meals'].select_dtypes(include='object').mode().iloc[0],inplace=True)
# data['Triceps skinfold thickness (mm)']=data['Triceps skinfold thickness (mm)'].fillna(data['Triceps skinfold thickness (mm)'].mean())






# 5.3 Analysing


# 6 Visualize =

# sns.relplot(x = countries_df['open_hours_per_week'], y = countries_df['avg_rating'], data = countries_df,
#  hue = countries_df['restaurants_count'])
# plt.show()
# x = countries_df['open_hours_per_week']
# y = countries_df['avg_rating']
# points_size = countries_df['restaurants_count']/100
# colors = countries_df['avg_rating']

# plt.scatter(x, y, s=points_size, c=colors)
# plt.title("Scatter Plot with increase in size of scatter points ", fontsize=12)
# plt.xlabel('x-axis', fontsize=12)
# plt.ylabel('y-axis', fontsize=12)
# plt.show()




# fig_2 = df['avg_rating'].plot(kind='hist')
# plt.xlabel("avg rating", labelpad=14)
# plt.title("Restaurant average rating")
# plt.show()

# 7 Create Insights


# data_1 = df[df['country'] == 'Italy']
# y = df['avg_rating']
# x = df['open_hours_per_week']
# hue
# print(y)
# print(x)
# sns.scatterplot(x, y , data = data_1, hue=)
# plt.show()

# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

# meal_list_final = set(flattened)
# print(meal_list_final)


#################### GRAVEYARD #######################################

# # meter uma func pra def key_word_split(df,col)
# data = list(df["meals"].fillna('Not listed').str.strip().str.split(","))
# flattened = [val for sublist in data for val in sublist]
# flattened_strip = []
# for word in flattened:
#     flattened_strip.extend(word.strip().split(','))
# meal_keyword_df = pd.Series(flattened_strip)
# print(meal_keyword_df.value_counts())


# fixing the issue of extra space after comma on some of the columns of the data frame
# df['meals'] = df['meals'].str.replace(", ", ",")
# df['cuisines'] = df['cuisines'].str.replace(", ", ",")
# df[','] = df['top_tags'].str.replace(", ", ",")
# df['awards'] = df['awards'].str.replace(", ", ",")
# df['features'] = df['features'].str.replace(", ", ",")


# df['vegan_options'] = df['vegan_options'].map({'Y': 1, 'N': 0})
# df['gluten_free'] = df['gluten_free'].map({'Y': 1, 'N': 0})
# df['special_diets'] = df['special_diets'].fillna(,inplace=True)
# df['special_diets'] = df.apply(
#    lambda row: row['vegetarian_friendly']*row['b'] if pd.isnull(row['c']) else row['special_diets'],
#    axis=
# print(df['meals'])
# print(df.head(10))

# print(df['region'].value_counts()[:20])
# print(df['province'].value_counts()[:20])

# print(df[df['restaurant_name']=="McDonald's"].head(10))

# df['avg_price']=df.groupby('city')['avg_price'].apply(lambda x:x.fillna(x.median()))
# print(df['avg_price'].isna().sum())
# df['avg_price']=df.groupby('region')['avg_price'].apply(lambda x:x.fillna(x.median()))
# print(df['avg_price'].isna().sum())
# df['avg_price']=df.groupby('province')['avg_price'].apply(lambda x:x.fillna(x.median()))
# print(df['avg_price'].isna().sum())
# df['avg_price']=df.groupby('country')['avg_price'].apply(lambda x:x.fillna(x.median()))
# print(df['avg_price'].isna().sum())

#vegan_df = df[df['vegan_options'] == 1].groupby('country').agg(
#    vegan_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size)).reset_index(level=0)
#gluten_free_df = df[df['gluten_free'] == 1].groupby('country').agg(
#    gluten_free_count=pd.NamedAgg(column='restaurant_link', aggfunc=np.size)).reset_index(level=0)