import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from xgboost import XGBRFClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import itertools as it

# data preparation
df_train = pd.read_csv(r'train.csv')

df_train['HomePlanet'].fillna('Earth', inplace=True)
df_train['Destination'].fillna('TRAPPIST-1e', inplace=True)
df_train['RoomService'].fillna(0, inplace=True)
df_train['FoodCourt'].fillna(0, inplace=True)
df_train['ShoppingMall'].fillna(0, inplace=True)
df_train['Spa'].fillna(0, inplace=True)
df_train['VRDeck'].fillna(0, inplace=True)
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)

df_train['Spent'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] \
                    + df_train['VRDeck']

df_train['Spent'] = df_train['Spent'].astype(bool)
df_train['CryoSleep'].fillna(-1, inplace=True)
df_train['CryoSleep'] = df_train['CryoSleep'].astype(float)

Cryo_index = df_train.query('CryoSleep == -1').CryoSleep.index
df_train['CryoSleep'].iloc[Cryo_index] = df_train['Spent'].iloc[Cryo_index]
df_train['CryoSleep'].iloc[Cryo_index] = df_train['CryoSleep'].iloc[Cryo_index] == False
df_train['CryoSleep'].iloc[Cryo_index] = df_train['CryoSleep'].iloc[Cryo_index].astype(int)
df_train['HomePlanet'].fillna('Earth', inplace=True)
df_train['Destination'].fillna('TRAPPIST-1e', inplace=True)
df_train['LuxSpent'] = df_train['RoomService'] + df_train['Spa'] + df_train['VRDeck']
df_train['VIP'].fillna(False, inplace=True)
df_train['Group'] = df_train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
group_dict = dict(df_train['Group'].value_counts() > 1)
df_train['withGroup'] = df_train['Group'].map(group_dict)
df_train['Family'] = df_train['Name'].astype(str).apply(lambda x: x.split(' ')[-1])
family_dict = dict(df_train['Family'].value_counts() > 1)
df_train['withFamily'] = df_train['Family'].map(family_dict)
df_train['Cabin'].fillna('F', inplace=True)
df_train['Deck'] = df_train['Cabin'].str.get(0)
df_train.drop(['PassengerId', 'Cabin', 'Name', 'Group', 'Family'], axis='columns', inplace=True)

HomePlanet = pd.get_dummies(df_train['HomePlanet'], prefix='HomePlanet')
Destination = pd.get_dummies(df_train['Destination'], prefix='Destination')
Deck = pd.get_dummies(df_train['Deck'], prefix='Deck')

data = df_train.join([HomePlanet, Destination, Deck])
data.drop(['HomePlanet', 'Destination', 'Deck'], axis='columns', inplace=True)
data['Spent'] = data['Spent'].astype(float)
data['LuxSpent'] = data['LuxSpent'].astype(float)
data['withGroup'] = data['withGroup'].astype(float)
data['withFamily'] = data['withFamily'].astype(float)
data['VIP'] = data['VIP'].astype(float)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
ind = pd.DataFrame(data.index)
ind['num'] = pd.Series(list(range(len(data))))
ind = ind.set_index(0)
data = data.join(ind).set_index('num')
X = data.join(pd.DataFrame(data_scaled,
                           columns=['t_Age', 't_RoomService', 't_FoodCourt', 't_ShoppingMall', 't_Spa', 't_VRDeck']))
X.drop(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis='columns', inplace=True)

y = X['Transported'].astype(int)
X['CryoSleep'] = X['CryoSleep'].astype(float)
X.drop('Transported', axis='columns', inplace=True)
# models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_xgb = XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=1000)
model_xgb.fit(X_train, y_train)

y_xgb_pred = model_xgb.predict(X_test)
print(classification_report(y_test, y_xgb_pred))
print(roc_auc_score(y_test, y_xgb_pred))

model_xgbrf = XGBRFClassifier(n_estimators=300, subsample=0.9, colsample_bynode=0.2)
model_xgbrf.fit(X_train, y_train)
y_xgbrf_pred = model_xgbrf.predict(X_test)
print(classification_report(y_test, y_xgbrf_pred))
print(roc_auc_score(y_test, y_xgbrf_pred))

model_svc = SVC(C=1.5, gamma='scale', kernel='rbf')
model_svc.fit(X_train, y_train)
y_svc_pred = model_svc.predict(X_test)
print(classification_report(y_test, y_svc_pred))
print(roc_auc_score(y_test, y_svc_pred))

hidden_layer = list(it.product([4, 5, 6], [4, 5, 6]))
tree_params = {'hidden_layer_sizes': hidden_layer,
               'activation': ['tanh', 'relu'],
               'solver': ['adam'],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'max_iter': [500, 1000]}
model_mlp = GridSearchCV(MLPClassifier(), tree_params, cv=5, scoring='accuracy', verbose=3)
model_mlp.fit(X_train.values, y_train.values)
y_mlp_pred = model_mlp.predict(X_test.values)
print(classification_report(y_test, y_mlp_pred))
