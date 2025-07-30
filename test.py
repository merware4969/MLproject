# # -----------------------------------------------------------
# # 1. 데이터 정의 (인종 변수 제외)
# # -----------------------------------------------------------
# num_cols = ['HOURSWRK', 'HEIGHT', 'WEIGHT', 'BMICALC']
# cat_cols = ['Occ_Group', 'Ind_Group', 'PAINARMS3M', 'PAINLEGS3M', 'HEALTH']

# X = df_cleaned[num_cols + cat_cols]
# y = df_cleaned['PAINBACK3M'].apply(lambda x: 0 if x == 1 else 1)

# # -----------------------------------------------------------
# # 2. 전처리
# # -----------------------------------------------------------
# preprocessor = ColumnTransformer([
#     ('num', StandardScaler(), num_cols),
#     ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
# ])

# # -----------------------------------------------------------
# # 3. 모델 사전 정의 (XGB 추가)
# # -----------------------------------------------------------
# models = {
#     'LogReg': LogisticRegression(max_iter=1000, solver='lbfgs'),
#     'RF': RandomForestClassifier(
#         n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
#     'LGBM': LGBMClassifier(
#         objective='binary', n_estimators=500, learning_rate=0.05,
#         num_leaves=31, random_state=42),
#     'XGB': XGBClassifier(
#         objective='binary:logistic', eval_metric='logloss',
#         n_estimators=500, learning_rate=0.05, max_depth=6,
#         subsample=0.8, colsample_bytree=0.8,
#         use_label_encoder=False, random_state=42,
#         n_jobs=-1)
# }

# # -----------------------------------------------------------
# # 4. 공통 파이프라인 + 5-fold CV
# # -----------------------------------------------------------
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scorers = {
#     'acc':  'accuracy',
#     'f1':   'f1',
#     'roc':  'roc_auc',
#     'prec': 'precision',
#     'rec':  'recall'
# }

# rows = []
# for name, clf in models.items():
#     pipe = ImbPipeline([
#         ('prep',  preprocessor),
#         ('smote', SMOTE(random_state=42)),
#         ('model', clf)
#     ])
#     scores = cross_validate(pipe, X, y, cv=cv,
#                             scoring=scorers, n_jobs=-1, return_train_score=False)
#     rows.append({
#         'Model':      name,
#         'Accuracy':   scores['test_acc'].mean(),
#         'F1':         scores['test_f1'].mean(),
#         'ROC_AUC':    scores['test_roc'].mean(),
#         'Precision':  scores['test_prec'].mean(),
#         'Recall':     scores['test_rec'].mean()
#     })

# results_df = pd.DataFrame(rows).set_index('Model').round(3)
# print(results_df)