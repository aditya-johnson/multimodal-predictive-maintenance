import pandas as pd

# ==========================
# STEP 1 — DEFINE COLUMN NAMES
# ==========================
columns = ['engine_id','cycle','setting1','setting2','setting3'] + \
          [f's{i}' for i in range(1,22)]

# ==========================
# STEP 2 — LOAD TRAIN
# ==========================
train = pd.read_csv(
    "data/train_FD001.txt",
    sep="\s+",
    header=None
)

train.columns = columns

print("\nTRAIN LOADED")
print(train.head())

# ==========================
# STEP 3 — CREATE TRAIN RUL
# ==========================
max_cycle = train.groupby('engine_id')['cycle'].max()

train['max_cycle'] = train['engine_id'].map(max_cycle)
train['RUL'] = train['max_cycle'] - train['cycle']

train.drop('max_cycle',axis=1,inplace=True)

print("\nTRAIN WITH RUL")
print(train[['engine_id','cycle','RUL']].head(20))

# ==========================
# STEP 4 — LOAD TEST
# ==========================
test = pd.read_csv(
    "data/test_FD001.txt",
    sep="\s+",
    header=None
)

test.columns = columns

print("\nTEST LOADED")
print(test.head())

# ==========================
# STEP 5 — LOAD TRUE RUL
# ==========================
rul = pd.read_csv(
    "data/RUL_FD001.txt",
    header=None
)

rul.columns=['RUL']

print("\nTRUE TEST RUL")
print(rul.head())

# ==========================
# STEP 6 — GET LAST CYCLE OF EACH ENGINE
# ==========================
last_cycle = test.groupby('engine_id')['cycle'].max()

test_last = test[test['cycle']==test['engine_id'].map(last_cycle)]

test_last = test_last.reset_index(drop=True)

# attach true RUL
test_last['RUL'] = rul['RUL']

# ==========================
# STEP 7 — SAVE CLEAN FILES
# ==========================
train.to_csv("train_clean.csv",index=False)
test_last.to_csv("test_clean.csv",index=False)

print("\nFILES SAVED:")
print("train_clean.csv")
print("test_clean.csv")
