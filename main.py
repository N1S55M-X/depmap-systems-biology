import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# ==========================================================
# LOAD
# ==========================================================

expr = pd.read_csv(
    "Expression_(Short-read)_Public_26Q1_subsetted.csv"
)

crispr = pd.read_csv(
    "CRISPR_Gene_Dependency_subsetted.csv"
)

expr.rename(
    columns={"Unnamed: 0":"DepMap_ID"},
    inplace=True
)

crispr.rename(
    columns={"Unnamed: 0":"DepMap_ID"},
    inplace=True
)

# ==========================================================
# MATCH CELL LINES
# ==========================================================

common = set(expr.DepMap_ID).intersection(
    set(crispr.DepMap_ID)
)

expr = expr[
    expr.DepMap_ID.isin(common)
]

crispr = crispr[
    crispr.DepMap_ID.isin(common)
]

expr = expr.sort_values(
    "DepMap_ID"
)

crispr = crispr.sort_values(
    "DepMap_ID"
)

# ==========================================================
# REMOVE BAD CRISPR GENES
# ==========================================================

bad_cols = crispr.columns[
    crispr.isna().mean() > 0.20
]

crispr = crispr.drop(
    columns=bad_cols
)

# ==========================================================
# FILL MISSING
# ==========================================================

expr_num = expr.drop(
    columns=["DepMap_ID"]
)

crispr_num = crispr.drop(
    columns=["DepMap_ID"]
)

expr_num = expr_num.fillna(
    expr_num.median()
)

crispr_num = crispr_num.fillna(
    crispr_num.median()
)

# ==========================================================
# NORMALIZE
# ==========================================================

expr_scaled = pd.DataFrame(
    StandardScaler().fit_transform(expr_num),
    columns=expr_num.columns,
    index=expr.DepMap_ID
)

crispr_scaled = pd.DataFrame(
    StandardScaler().fit_transform(crispr_num),
    columns=crispr_num.columns,
    index=crispr.DepMap_ID
)

# ==========================================================
# EXPRESSION MODULES
# ==========================================================

expr_gene_matrix = expr_scaled.T

expr_kmeans = KMeans(
    n_clusters=50,
    random_state=42,
    n_init=20
)

expr_labels = expr_kmeans.fit_predict(
    expr_gene_matrix
)

expr_cluster_df = pd.DataFrame({
    "Gene":expr_gene_matrix.index,
    "Cluster":expr_labels
})

expr_modules = pd.DataFrame(
    index=expr_scaled.index
)

for c in sorted(
    expr_cluster_df.Cluster.unique()
):

    genes = expr_cluster_df[
        expr_cluster_df.Cluster == c
    ].Gene

    expr_modules[
        f"Expr_Module_{c}"
    ] = expr_scaled[
        genes
    ].mean(axis=1)

# ==========================================================
# CRISPR MODULES
# ==========================================================

crispr_gene_matrix = crispr_scaled.T

crispr_kmeans = KMeans(
    n_clusters=50,
    random_state=42,
    n_init=20
)

crispr_labels = crispr_kmeans.fit_predict(
    crispr_gene_matrix
)

crispr_cluster_df = pd.DataFrame({
    "Gene":crispr_gene_matrix.index,
    "Cluster":crispr_labels
})

crispr_modules = pd.DataFrame(
    index=crispr_scaled.index
)

for c in sorted(
    crispr_cluster_df.Cluster.unique()
):

    genes = crispr_cluster_df[
        crispr_cluster_df.Cluster == c
    ].Gene

    crispr_modules[
        f"CRISPR_Module_{c}"
    ] = crispr_scaled[
        genes
    ].mean(axis=1)

# ==========================================================
# MODULE LINKS
# ==========================================================

combined = pd.concat(
    [expr_modules, crispr_modules],
    axis=1
)

corr = combined.corr()

links = []

for e in expr_modules.columns:

    for c in crispr_modules.columns:

        links.append([
            e,
            c,
            corr.loc[e,c]
        ])

links = pd.DataFrame(
    links,
    columns=[
        "Expression_Module",
        "CRISPR_Module",
        "Correlation"
    ]
)

links = links.sort_values(
    "Correlation",
    ascending=False
)

# ==========================================================
# PREDICT MODULES
# ==========================================================

results = []

for target in crispr_modules.columns:

    X = expr_modules

    y = crispr_modules[target]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        target,
        r2
    ])

results = pd.DataFrame(
    results,
    columns=[
        "CRISPR_Module",
        "R2"
    ]
)

results = results.sort_values(
    "R2",
    ascending=False
)

# ==========================================================
# SAVE
# ==========================================================

expr_cluster_df.to_csv(
    "Expression_Gene_Modules.csv",
    index=False
)

crispr_cluster_df.to_csv(
    "CRISPR_Gene_Modules.csv",
    index=False
)

links.to_csv(
    "Expression_CRISPR_Module_Links.csv",
    index=False
)

results.to_csv(
    "CRISPR_Module_Prediction_Results.csv",
    index=False
)

print(results.head(10))
