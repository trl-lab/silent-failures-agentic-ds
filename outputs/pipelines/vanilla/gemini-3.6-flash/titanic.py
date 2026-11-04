import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/titanic/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/titanic/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

train_len = len(train)
y_train = train['Survived'].values.astype(int)

df = pd.concat([train.drop(columns=['Survived']), test], ignore_index=True)

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
}
df['Title'] = df['Title'].map(lambda x: title_map.get(x, 'Rare'))

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
df['HasCabin'] = df['Cabin'].notna().astype(int)

df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
df['LogFare'] = np.log1p(df['Fare'])

df['Embarked'] = df['Embarked'].fillna('S')

df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())

df['IsChild'] = ((df['Age'] < 14) | (df['Title'] == 'Master')).astype(int)
df['IsFemale'] = (df['Sex'] == 'female').astype(int)
df['IsWc'] = ((df['Sex'] == 'female') | (df['Age'] < 14) | (df['Title'] == 'Master')).astype(int)

df['AgeClass'] = df['Age'] * df['Pclass']
df['TicketGroupSize'] = df['Ticket'].map(df['Ticket'].value_counts())

df['Surname'] = df['Name'].str.split(',').str[0].str.strip()
df['Surname_Pclass'] = df['Surname'] + '_' + df['Pclass'].astype(str)

df['Survived_Temp'] = np.nan
df.loc[:train_len - 1, 'Survived_Temp'] = y_train

train_indices = list(range(train_len))

def calc_group_surv(df, group_col):
    res = pd.Series(np.nan,