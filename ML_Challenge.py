# ML Challenge - Property Value Prediction
# Author: Fabio Pugno
#
# Note: I used Claude (Anthropic) as a Generative AI tool to help with parts
# of the code in this script, in accordance with the assignment guidelines.
# The full record of my interaction with the AI has been submitted via the webform.
#
# ── Method summary ────────────────────────────────────────────────────────────
# The target metric is MAE (in euros). Two families of gradient-boosted models
# are trained (XGBoost, LightGBM, CatBoost), all with an MAE-aligned objective:
#   1. SPECIALIZED models: one model per property category, trained in both
#      log-target and raw-target space.
#   2. GLOBAL models: a single model over all categories at once (category passed
#      as a feature). Pooling data helps the small, heavy-tailed categories
#      (activite, bati_mixte, volume) that have too few records on their own.
# Per category, predictions are blended with weights selected by honest 5-fold
# out-of-fold cross-validation and verified on a held-out split to ensure the
# blend generalizes (heavy-tailed categories use robust fixed equal weights;
# large categories use optimized weights). See report for the validation table.

import json
import zipfile
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

RNG = 42

# ── Load data ──────────────────────────────────────────────────────────────────
with open('train.json') as f:
    train_raw = json.load(f)
with open('test.json') as f:
    test_raw = json.load(f)

df_train_full = pd.DataFrame(train_raw)
df_test       = pd.DataFrame(test_raw)
TARGET = 'property_value'

# ── Property type → model category ────────────────────────────────────────────
CAT_MAP = {
    'UNE MAISON': 'house', 'DES MAISONS': 'house', 'MAISON - INDETERMINEE': 'house',
    'UN APPARTEMENT': 'apartment', 'DEUX APPARTEMENTS': 'apartment', 'APPARTEMENT INDETERMINE': 'apartment',
    'TERRAIN DE TYPE TERRE ET PRE': 'land', 'TERRAIN DE TYPE TAB': 'land', 'TERRAIN ARTIFICIALISE MIXTE': 'land',
    "TERRAIN D'AGREMENT": 'land', 'TERRAIN FORESTIER': 'land', 'TERRAIN NON BATIS INDETERMINE': 'land',
    'TERRAIN VITICOLE': 'land', 'TERRAIN LANDES ET EAUX': 'land', 'TERRAIN VERGER': 'land',
    'TERRAIN NATUREL MIXTE': 'land', "TERRAIN D'EXTRACTION": 'land', 'TERRAIN AGRICOLE MIXTE': 'land',
    'TERRAIN DE TYPE RESEAU': 'land',
    'BATI - INDETERMINE : Vefa sans descriptif': 'vefa',
    'BATI - INDETERMINE : Vente avec volume(s)': 'volume',
    'UNE DEPENDANCE': 'dependency', 'DES DEPENDANCES': 'dependency',
    'ACTIVITE': 'activite',
    'BATI MIXTE - LOGEMENT/ACTIVITE': 'bati_mixte', 'BATI MIXTE - LOGEMENTS': 'bati_mixte',
}
get_cat = lambda x: CAT_MAP.get(x, 'land')
df_train_full['category'] = df_train_full['property_type'].apply(get_cat)
df_test['category']       = df_test['property_type'].apply(get_cat)

# Remove non-market transactions (symbolic transfers, gifts, etc.)
df_train_full = df_train_full[df_train_full[TARGET] >= 1000].copy().reset_index(drop=True)
df_test.reset_index(drop=True, inplace=True)

# ── Feature engineering ────────────────────────────────────────────────────────
APT_ROOM   = ['num_apt_1_room','num_apt_2_rooms','num_apt_3_rooms','num_apt_4_rooms','num_apt_5plus_rooms']
HOUSE_ROOM = ['num_house_1_room','num_house_2_rooms','num_house_3_rooms','num_house_4_rooms','num_house_5plus_rooms']
APT_AREA   = ['area_apt_1_room','area_apt_2_rooms','area_apt_3_rooms','area_apt_4_rooms','area_apt_5plus_rooms']
HOUSE_AREA = ['area_house_1_room','area_house_2_rooms','area_house_3_rooms','area_house_4_rooms','area_house_5plus_rooms']

def add_features(df):
    d = df.copy()
    d['commune_first']   = d['commune_codes'].astype(str).str.split().str[0]
    d['cadastral_first'] = d['cadastral_sections'].astype(str).str.split().str[0]
    d['commune_section'] = d['commune_first'] + '_' + d['cadastral_first']
    d['apt_rooms']   = d[APT_ROOM].fillna(0).sum(axis=1)
    d['house_rooms'] = d[HOUSE_ROOM].fillna(0).sum(axis=1)
    d['total_rooms'] = d['apt_rooms'] + d['house_rooms']
    d['apt_area_sum']   = d[APT_AREA].fillna(0).sum(axis=1)
    d['house_area_sum'] = d[HOUSE_AREA].fillna(0).sum(axis=1)
    apt_d = d['apt_area_sum'].clip(lower=1)
    d['apt_weighted_rooms'] = (1*d['area_apt_1_room'].fillna(0)+2*d['area_apt_2_rooms'].fillna(0)+
        3*d['area_apt_3_rooms'].fillna(0)+4*d['area_apt_4_rooms'].fillna(0)+5*d['area_apt_5plus_rooms'].fillna(0))/apt_d
    d.loc[d['apt_area_sum']==0,'apt_weighted_rooms']=0
    h_d = d['house_area_sum'].clip(lower=1)
    d['house_weighted_rooms'] = (1*d['area_house_1_room'].fillna(0)+2*d['area_house_2_rooms'].fillna(0)+
        3*d['area_house_3_rooms'].fillna(0)+4*d['area_house_4_rooms'].fillna(0)+5*d['area_house_5plus_rooms'].fillna(0))/h_d
    d.loc[d['house_area_sum']==0,'house_weighted_rooms']=0
    d['total_built'] = d['built_area'].fillna(0)+d['house_area'].fillna(0)+d['apartment_area'].fillna(0)
    d['area_per_room'] = d['total_built']/(d['total_rooms']+1)
    d['apt_area_per_room'] = d['apartment_area'].fillna(0)/(d['apt_rooms']+1)
    d['house_area_per_room'] = d['house_area'].fillna(0)/(d['house_rooms']+1)
    d['land_area_per_parcel'] = d['land_area'].fillna(0)/(d['num_parcels']+1)
    for col in (['land_area','built_area','house_area','apartment_area','total_built',
                 'land_area_per_parcel','apt_area_per_room','house_area_per_room']+APT_AREA+HOUSE_AREA):
        d[f'log_{col}'] = np.log1p(d[col].fillna(0))
    d['has_built'] = (d['total_built']>0).astype(int)
    d['has_land']  = (d['land_area'].fillna(0)>0).astype(int)
    d['future_sale'] = d['future_sale'].astype(int)
    d['multi_commune'] = (d['num_communes']>1).astype(int)
    return d

df_train_full = add_features(df_train_full)
df_test       = add_features(df_test)

# Global (low-cardinality) target encodings — median of log target
def te_global(tr, te, col):
    log_y = np.log1p(tr[TARGET].values); gm = np.median(log_y)
    meds = pd.Series(log_y).groupby(pd.Series(tr[col].values)).median()
    return (pd.Series(tr[col].values).map(meds).fillna(gm).values,
            pd.Series(te[col].values).map(meds).fillna(gm).values)

for c in ['transaction_type', 'property_type']:
    a, b = te_global(df_train_full, df_test, c)
    df_train_full[f'te_{c}'] = a
    df_test[f'te_{c}'] = b

LOG_APT_AREA   = [f'log_{c}' for c in APT_AREA]
LOG_HOUSE_AREA = [f'log_{c}' for c in HOUSE_AREA]
BASE = ['year','month','future_sale','multi_commune','num_lots','te_transaction_type','te_property_type','has_built','has_land']

# Per-category specialized feature sets
CATFEAT = {
 'house': BASE+['total_built','log_total_built','house_area','log_house_area','land_area','log_land_area',
   'house_rooms','house_weighted_rooms','area_per_room','house_area_per_room','log_house_area_per_room',
   'num_houses','num_dependencies','num_sections','te_commune_first','te_commune_section']+LOG_HOUSE_AREA,
 'apartment': BASE+['total_built','log_total_built','apartment_area','log_apartment_area','apt_rooms','apt_weighted_rooms',
   'area_per_room','apt_area_per_room','log_apt_area_per_room','num_apartments','num_dependencies',
   'te_commune_first','te_commune_section']+LOG_APT_AREA,
 'land': BASE+['land_area','log_land_area','land_area_per_parcel','log_land_area_per_parcel','total_built','log_total_built',
   'num_parcels','num_sections','num_premises','te_commune_first','te_commune_section'],
 'vefa': BASE+['total_built','log_total_built','land_area','log_land_area','total_rooms','area_per_room',
   'apt_rooms','apt_weighted_rooms','apt_area_per_room','house_rooms','house_area_per_room',
   'num_apartments','num_houses','te_commune_first','te_commune_section']+LOG_APT_AREA,
}
CB_CAT = ['commune_first','commune_section','property_type','transaction_type']
CATFEAT_CB = {c:[f for f in feats if f not in ('te_commune_first','te_commune_section','te_transaction_type','te_property_type')]+CB_CAT
              for c,feats in CATFEAT.items()}

# Global model feature set (union of useful features; category passed as cat feature for CB)
GLOBAL_NUM = ['year','month','future_sale','multi_commune','num_lots','num_communes','num_sections','num_parcels',
  'num_premises','num_houses','num_apartments','num_dependencies','num_commercial',
  'te_transaction_type','te_property_type','has_built','has_land',
  'total_built','log_total_built','land_area','log_land_area','built_area','log_built_area',
  'house_area','log_house_area','apartment_area','log_apartment_area',
  'total_rooms','apt_rooms','house_rooms','area_per_room','apt_area_per_room','house_area_per_room',
  'land_area_per_parcel','log_land_area_per_parcel','apt_weighted_rooms','house_weighted_rooms',
  'te_commune_first','te_commune_section'] + LOG_APT_AREA + LOG_HOUSE_AREA
GLOBAL_CB_CATS = ['commune_first','commune_section','property_type','transaction_type','category']
GLOBAL_CB = [c for c in GLOBAL_NUM if c not in ('te_commune_first','te_commune_section','te_transaction_type','te_property_type')] + GLOBAL_CB_CATS

# ── Model parameters (MAE-aligned objectives) ─────────────────────────────────
XGB  = dict(n_estimators=3000, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0, objective='reg:absoluteerror',
            random_state=RNG, n_jobs=-1, early_stopping_rounds=100, eval_metric='mae')
LGBM = dict(n_estimators=3000, learning_rate=0.02, num_leaves=63, max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.3, reg_lambda=1.0, objective='regression_l1',
            random_state=RNG, n_jobs=-1, verbose=-1)
CB   = dict(iterations=3000, learning_rate=0.02, depth=8, loss_function='MAE', eval_metric='MAE',
            subsample=0.8, rsm=0.8, random_seed=RNG, thread_count=-1, verbose=0, early_stopping_rounds=100)

# ── Per-category blend recipe (validated by 5-fold OOF + held-out generalization test) ──
# Component keys: spec models {xgb,lgbm,cb}×{log,raw}; global models {g_xgb,g_lgbm,g_cb}.
# Large categories use OOF-optimized weights; heavy-tailed small categories use robust
# fixed equal weights on the global models (weight optimization overfits there).
RECIPE = {
    'house':      {'lgbm_log':0.05,'cb_log':0.10,'lgbm_raw':0.08,'cb_raw':0.24,'g_lgbm':0.13,'g_cb':0.40},
    'apartment':  {'cb_log':0.05,'lgbm_raw':0.32,'g_lgbm':0.25,'g_cb':0.38},
    'land':       {'lgbm_log':0.06,'cb_log':0.55,'cb_raw':0.39},
    'vefa':       {'xgb_log':0.88,'cb_raw':0.12},
    'dependency': {'g_xgb':0.20,'g_lgbm':0.05,'g_cb':0.75},
    'activite':   {'g_xgb':1/3,'g_lgbm':1/3,'g_cb':1/3},
    'bati_mixte': {'g_xgb':1/3,'g_lgbm':1/3,'g_cb':1/3},
    'volume':     {'g_xgb':1/3,'g_lgbm':1/3,'g_cb':1/3},
}

def fit_predict(kind, mode, Xtr, ytr_euro, Xte, cat_features=None, seeds=(RNG,)):
    """Train one model with early-stopping on an internal holdout to pick the
    iteration count, then refit on all rows and average over `seeds` (bagging:
    each seed reshuffles subsample/colsample, reducing variance). Returns the
    mean euro-space test prediction and the chosen iteration count."""
    yt = np.log1p(ytr_euro) if mode == 'log' else ytr_euro.astype(float)
    inv = (lambda p: np.expm1(p)) if mode == 'log' else (lambda p: p)
    itr, iva = train_test_split(np.arange(len(Xtr)), test_size=0.15, random_state=RNG)
    Xa, Xb, ya, yb = Xtr.iloc[itr], Xtr.iloc[iva], yt[itr], yt[iva]

    if kind == 'xgb':
        m = XGBRegressor(**XGB); m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        n = int(m.best_iteration * 1.1) + 1
        preds = []
        for s in seeds:
            mf = XGBRegressor(**{**XGB, 'early_stopping_rounds': None, 'n_estimators': n, 'random_state': s})
            mf.fit(Xtr, yt); preds.append(inv(mf.predict(Xte)))
        return np.mean(preds, axis=0), n
    if kind == 'lgbm':
        m = LGBMRegressor(**LGBM)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        n = int(m.best_iteration_ * 1.1) + 1
        preds = []
        for s in seeds:
            mf = LGBMRegressor(**{**LGBM, 'n_estimators': n, 'random_state': s})
            mf.fit(Xtr, yt); preds.append(inv(mf.predict(Xte)))
        return np.mean(preds, axis=0), n
    # catboost
    m = CatBoostRegressor(**CB); m.fit(Xa, ya, eval_set=(Xb, yb), cat_features=cat_features, verbose=False)
    n = int(m.best_iteration_ * 1.1) + 1
    preds = []
    for s in seeds:
        mf = CatBoostRegressor(**{**CB, 'early_stopping_rounds': None, 'iterations': n, 'random_seed': s})
        mf.fit(Xtr, yt, cat_features=cat_features, verbose=False); preds.append(inv(mf.predict(Xte)))
    return np.mean(preds, axis=0), n

clip = lambda v: np.clip(v, 0, None)

# ── Phase A: GLOBAL models (trained on all categories, predict all test rows) ──
print("Phase A: Global models")
print("-" * 60)
df_full = df_train_full.copy()
dft = df_test.copy()

# Global commune target-encoding (pooled over all categories) via KFold to avoid leakage
for col in ['commune_first', 'commune_section']:
    enc = np.zeros(len(df_full)); ly = np.log1p(df_full[TARGET].values); gm = np.median(ly)
    kf = KFold(5, shuffle=True, random_state=RNG)
    for tri, vai in kf.split(df_full):
        meds = pd.Series(ly[tri]).groupby(pd.Series(df_full[col].values[tri])).median()
        enc[vai] = pd.Series(df_full[col].values[vai]).map(meds).fillna(gm).values
    df_full[f'te_{col}'] = enc
    full_meds = pd.Series(ly).groupby(pd.Series(df_full[col].values)).median()
    dft[f'te_{col}'] = pd.Series(dft[col].values).map(full_meds).fillna(gm).values

# Bag the global models over several seeds: variance reduction directly benefits
# the heavy-tailed categories (activite, bati_mixte, volume) that dominate the MAE.
GLOBAL_SEEDS = (42, 1, 7)
g_preds = {}
for kind in ['xgb', 'lgbm', 'cb']:
    if kind == 'cb':
        Xtr = df_full[GLOBAL_CB].fillna(0).copy(); Xte = dft[GLOBAL_CB].fillna(0).copy()
        for c in GLOBAL_CB_CATS: Xtr[c] = Xtr[c].astype(str); Xte[c] = Xte[c].astype(str)
        p, n = fit_predict('cb', 'log', Xtr, df_full[TARGET].values, Xte,
                           cat_features=GLOBAL_CB_CATS, seeds=GLOBAL_SEEDS)
    else:
        Xtr = df_full[GLOBAL_NUM].fillna(0); Xte = dft[GLOBAL_NUM].fillna(0)
        p, n = fit_predict(kind, 'log', Xtr, df_full[TARGET].values, Xte, seeds=GLOBAL_SEEDS)
    g_preds[f'g_{kind}'] = clip(p)
    print(f"  global {kind:4s}  n_iter={n}  (bagged x{len(GLOBAL_SEEDS)})")

# ── Phase B: SPECIALIZED models for categories that need them ─────────────────
print("\nPhase B: Specialized models")
print("-" * 60)
# Which (model, mode) specialized components each category needs (nonzero recipe weight)
SPEC_NEEDED = {
    'house':     [('lgbm','log'),('cb','log'),('lgbm','raw'),('cb','raw')],
    'apartment': [('cb','log'),('lgbm','raw')],
    'land':      [('lgbm','log'),('cb','log'),('cb','raw')],
    'vefa':      [('xgb','log'),('cb','raw')],
}
spec_preds = {}  # cat -> {component_key: euro preds on that cat's test rows}
for cat, needed in SPEC_NEEDED.items():
    m_full = df_full['category'] == cat
    m_test = dft['category'] == cat
    sub = df_full[m_full].reset_index(drop=True)
    sub_te = dft[m_test].reset_index(drop=True)
    # Within-category commune target-encoding (KFold on train, full-median to test)
    for col in ['commune_first', 'commune_section']:
        enc = np.zeros(len(sub)); ly = np.log1p(sub[TARGET].values); gm = np.median(ly)
        for tri, vai in KFold(5, shuffle=True, random_state=RNG).split(sub):
            meds = pd.Series(ly[tri]).groupby(pd.Series(sub[col].values[tri])).median()
            enc[vai] = pd.Series(sub[col].values[vai]).map(meds).fillna(gm).values
        sub[f'te_{col}'] = enc
        fm = pd.Series(ly).groupby(pd.Series(sub[col].values)).median()
        sub_te[f'te_{col}'] = pd.Series(sub_te[col].values).map(fm).fillna(gm).values
    feats = list(dict.fromkeys(CATFEAT[cat]))
    feats_cb = list(dict.fromkeys(CATFEAT_CB[cat]))
    spec_preds[cat] = {}
    y_cat = sub[TARGET].values
    for kind, mode in needed:
        if kind == 'cb':
            Xtr = sub[feats_cb].fillna(0).copy(); Xte = sub_te[feats_cb].fillna(0).copy()
            for c in CB_CAT: Xtr[c] = Xtr[c].astype(str); Xte[c] = Xte[c].astype(str)
            p, n = fit_predict('cb', mode, Xtr, y_cat, Xte, cat_features=CB_CAT)
        else:
            Xtr = sub[feats].fillna(0); Xte = sub_te[feats].fillna(0)
            p, n = fit_predict(kind, mode, Xtr, y_cat, Xte)
        spec_preds[cat][f'{kind}_{mode}'] = clip(p)
    print(f"  [{cat:11s}] n={m_full.sum():5d}  trained {len(needed)} specialized components")

# ── Phase C: Blend per recipe ─────────────────────────────────────────────────
print("\nPhase C: Blend & write predictions")
print("-" * 60)
test_preds = pd.Series(np.nan, index=dft.index)
for cat, weights in RECIPE.items():
    m_test = (dft['category'] == cat).values
    if m_test.sum() == 0:
        continue
    blend = np.zeros(m_test.sum())
    wsum = 0.0
    for comp, w in weights.items():
        if comp.startswith('g_'):
            comp_pred = g_preds[comp][m_test]
        else:
            comp_pred = spec_preds[cat][comp]
        blend += w * comp_pred
        wsum += w
    test_preds[m_test] = blend / wsum   # normalize in case weights don't sum exactly to 1
    print(f"  [{cat:11s}]  n_test={m_test.sum():5d}  components={list(weights.keys())}")

fallback = df_full[TARGET].median()
final = test_preds.fillna(fallback).clip(lower=0)
predictions = [{'property_value': float(v)} for v in final]

with open('predicted.json', 'w') as f:
    json.dump(predictions, f)
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('predicted.json')

print(f"\nDone. Generated predicted.json ({len(predictions)} predictions) and submission.zip.")
print(f"Pred stats — min={final.min():.0f}  median={final.median():.0f}  max={final.max():.0f}")
print("Upload submission.zip to Codabench.")
