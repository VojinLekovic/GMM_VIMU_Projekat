import os
import re
import ast
import argparse
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


QUANT_RE = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)?\s*([A-Za-zčćžšđČĆŽŠĐ]+)?\s*$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", required=True)
    parser.add_argument("--shops", required=True)
    parser.add_argument("--users", required=True)
    parser.add_argument("--out_dir", default="results")
    return parser.parse_args()


def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip()
    if value in ("", "nan", "None", "null"):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [part.strip(" '\"") for part in value.split(",") if part.strip(" '\"")]


def first_nonnull(series):
    series = series.dropna()
    return series.iloc[0] if len(series) else np.nan


def parse_quantity(quantity):
    if pd.isna(quantity):
        return np.nan, "unknown"

    quantity = str(quantity).strip()
    if quantity == "":
        return np.nan, "unknown"

    match = QUANT_RE.match(quantity)
    if not match:
        return np.nan, "unknown"

    value, unit = match.groups()
    unit = (unit or "unknown").lower()
    unit = {"kg": "kg", "g": "g", "l": "l", "ml": "ml", "kom": "kom"}.get(unit, unit)

    if value is None:
        return np.nan, unit

    return float(value.replace(",", ".")), unit


def convert_to_base(value, unit):
    if pd.isna(value):
        return np.nan
    if unit == "kg":
        return value * 1000.0
    if unit == "g":
        return value
    if unit == "l":
        return value * 1000.0
    if unit == "ml":
        return value
    if unit == "kom":
        return value
    return np.nan


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s/-]{6,}", " ", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\sčćžšđ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_analytical_dataset(products_path, shops_path, users_path):
    products = pd.read_csv(products_path, encoding="utf-8-sig")
    shops = pd.read_csv(shops_path, encoding="utf-8-sig")
    users = pd.read_csv(users_path, encoding="utf-8-sig")

    for col in ["deliveryCities", "deliveryRegions", "images", "allergens"]:
        products[col] = products[col].fillna("[]")
        products[f"{col}_parsed"] = products[col].apply(parse_list)

    shops["followers"] = shops["followers"].fillna("[]")
    shops["followers_parsed"] = shops["followers"].apply(parse_list)

    users["savedProducts"] = users["savedProducts"].fillna("[]")
    users["savedProducts_parsed"] = users["savedProducts"].apply(parse_list)

    for dataset, cols in (
        (products, ["createdAt", "editedAt"]),
        (shops, ["createdAt"]),
        (users, ["createdAt"]),
    ):
        for col in cols:
            dataset[f"{col}_dt"] = pd.to_datetime(dataset[col], utc=True, errors="coerce")

    saved_counter = Counter()
    for product_ids in users["savedProducts_parsed"]:
        saved_counter.update(product_ids)

    products["saved_count"] = products["document_id"].map(
        lambda doc_id: saved_counter.get(doc_id, 0)
    ).astype(int)
    shops["followers_count"] = shops["followers_parsed"].apply(len).astype(int)

    products[["quantity_value", "quantity_unit"]] = products["quantity"].apply(
        lambda value: pd.Series(parse_quantity(value))
    )
    products["quantity_base"] = [
        convert_to_base(value, unit)
        for value, unit in zip(products["quantity_value"], products["quantity_unit"])
    ]

    quantity_nonnull = products["quantity_base"].dropna()
    clip_hi = quantity_nonnull.quantile(0.95) if not quantity_nonnull.empty else np.nan
    products["quantity_base_clipped"] = products["quantity_base"].clip(upper=clip_hi)
    products["quantity_log"] = np.log1p(products["quantity_base_clipped"])

    products["text_clean"] = (
        products["name"].fillna("") + " " + products["description"].fillna("")
    ).apply(clean_text)
    products["delivery_cities_count"] = products["deliveryCities_parsed"].apply(len)
    products["delivery_regions_count"] = products["deliveryRegions_parsed"].apply(len)
    products["images_count"] = products["images_parsed"].apply(len)
    products["allergens_count"] = products["allergens_parsed"].apply(len)
    products["description_len"] = products["description"].fillna("").str.len()
    products["name_len"] = products["name"].fillna("").str.len()
    products["has_description"] = (
        products["description"].fillna("").str.strip() != ""
    ).astype(int)
    products["edited_flag"] = products["editedAt_dt"].notna().astype(int)
    products["edit_delay_hours"] = (
        (products["editedAt_dt"] - products["createdAt_dt"]).dt.total_seconds() / 3600.0
    ).clip(lower=0)

    snapshot_candidates = [
        products["createdAt_dt"].max(),
        shops["createdAt_dt"].max(),
        users["createdAt_dt"].max(),
        products["editedAt_dt"].max(),
    ]
    snapshot_candidates = [ts for ts in snapshot_candidates if pd.notna(ts)]
    snapshot_time = max(snapshot_candidates) if snapshot_candidates else pd.Timestamp.utcnow()

    products["listing_age_days"] = (
        (snapshot_time - products["createdAt_dt"]).dt.total_seconds() / 86400.0
    )

    shop_agg = (
        shops.groupby("sellerId", as_index=False)
        .agg(
            city=("city", first_nonnull),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
            shop_createdAt=("createdAt_dt", "min"),
            followers_count=("followers_count", "max"),
        )
    )

    seller_prod_agg = (
        products.groupby("sellerId", as_index=False)
        .agg(
            seller_product_count=("document_id", "nunique"),
            seller_avg_price=("price", "mean"),
            seller_category_diversity=("category", "nunique"),
        )
    )

    df = products.merge(shop_agg, on="sellerId", how="left").merge(
        seller_prod_agg, on="sellerId", how="left"
    )

    df["shop_age_days"] = (
        (snapshot_time - df["shop_createdAt"]).dt.total_seconds() / 86400.0
    )
    city_counts = df["city"].fillna("Nepoznato").value_counts()
    df["city"] = df["city"].fillna("Nepoznato")
    df["city_grouped"] = df["city"].where(df["city"].map(city_counts) >= 3, "Ostalo")
    df["price_log"] = np.log1p(pd.to_numeric(df["price"], errors="coerce"))
    df["description_len_log"] = np.log1p(df["description_len"])

    edit_delay_nonnull = df["edit_delay_hours"].dropna()
    edit_delay_hi = edit_delay_nonnull.quantile(0.95) if not edit_delay_nonnull.empty else 0.0
    df["edit_delay_hours_filled"] = df["edit_delay_hours"].fillna(0).clip(upper=edit_delay_hi)
    df["seller_avg_price_log"] = np.log1p(df["seller_avg_price"])
    df["pickupAvailable"] = df["pickupAvailable"].astype(int)

    return df


def build_feature_spaces(df):
    numeric_cols = [
        "price_log",
        "quantity_log",
        "description_len_log",
        "name_len",
        "delivery_cities_count",
        "delivery_regions_count",
        "images_count",
        "allergens_count",
        "listing_age_days",
        "edit_delay_hours_filled",
        "lat",
        "lng",
        "seller_product_count",
        "seller_avg_price_log",
        "seller_category_diversity",
        "shop_age_days",
    ]
    binary_cols = ["pickupAvailable", "has_description", "edited_flag"]
    cat_cols = ["category", "deliveryType", "city_grouped", "quantity_unit"]

    pre_struct = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            ("bin", "passthrough", binary_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_struct = pre_struct.fit_transform(df)

    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df["text_clean"])

    svd = TruncatedSVD(n_components=5, random_state=42)
    X_text = StandardScaler().fit_transform(svd.fit_transform(X_tfidf))

    X_struct_text = np.hstack([X_struct, X_text])

    pca_struct = PCA(n_components=6, random_state=42)
    X_struct_pca = pca_struct.fit_transform(X_struct)

    pca_struct_text = PCA(n_components=6, random_state=42)
    X_struct_text_pca = pca_struct_text.fit_transform(X_struct_text)

    return {
        "pre_struct": pre_struct,
        "tfidf": tfidf,
        "svd": svd,
        "pca_struct": pca_struct,
        "pca_struct_text": pca_struct_text,
        "X_struct": X_struct,
        "X_struct_text": X_struct_text,
        "X_struct_pca": X_struct_pca,
        "X_struct_text_pca": X_struct_text_pca,
    }


def evaluate_gmm_grid(feature_spaces):
    candidates = {
        "structured_pca6": feature_spaces["X_struct_pca"],
        "structured_text_pca6": feature_spaces["X_struct_text_pca"],
    }
    rows = []

    for name, X in candidates.items():
        for covariance in ["full", "tied", "diag", "spherical"]:
            for k in range(2, 7):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance,
                    reg_covar=1e-4,
                    n_init=5,
                    max_iter=200,
                    init_params="kmeans",
                    random_state=42,
                )
                gmm.fit(X)
                labels = gmm.predict(X)
                silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
                probs = gmm.predict_proba(X)
                entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean()

                rows.append(
                    {
                        "feature_set": name,
                        "covariance": covariance,
                        "k": k,
                        "bic": gmm.bic(X),
                        "aic": gmm.aic(X),
                        "loglik": gmm.score(X),
                        "silhouette": silhouette,
                        "entropy": entropy,
                        "min_cluster_size": pd.Series(labels).value_counts().min(),
                    }
                )

    return pd.DataFrame(rows).sort_values(["feature_set", "bic", "aic"])


def validate_candidate(X, k, covariance, n_splits=10):
    splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    loglik_scores = []
    bic_scores = []
    silhouette_scores = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X)):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            reg_covar=1e-4,
            n_init=3,
            max_iter=200,
            init_params="kmeans",
            random_state=100 + i,
        )
        gmm.fit(X[train_idx])
        loglik_scores.append(gmm.score(X[test_idx]))
        bic_scores.append(gmm.bic(X[train_idx]))
        labels = gmm.predict(X[train_idx])
        silhouette_scores.append(
            silhouette_score(X[train_idx], labels) if len(set(labels)) > 1 else np.nan
        )

    return {
        "cv_test_loglik_mean": float(np.mean(loglik_scores)),
        "cv_test_loglik_std": float(np.std(loglik_scores)),
        "cv_train_bic_mean": float(np.mean(bic_scores)),
        "cv_train_bic_std": float(np.std(bic_scores)),
        "cv_sil_mean": float(np.nanmean(silhouette_scores)),
    }


def bootstrap_stability(X, k, covariance, n_boot=10):
    rng = np.random.RandomState(42)
    labelings = []

    for b in range(n_boot):
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            reg_covar=1e-4,
            n_init=2,
            max_iter=150,
            init_params="kmeans",
            random_state=200 + b,
        )
        gmm.fit(X[boot_idx])
        labelings.append(gmm.predict(X))

    ari_scores = []
    nmi_scores = []

    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            ari_scores.append(adjusted_rand_score(labelings[i], labelings[j]))
            nmi_scores.append(normalized_mutual_info_score(labelings[i], labelings[j]))

    return {
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
    }


def fit_final_model(df, X, k=6, covariance="spherical"):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance,
        reg_covar=1e-4,
        n_init=10,
        max_iter=300,
        init_params="kmeans",
        random_state=42,
    )
    gmm.fit(X)

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    segmented = df.copy()
    segmented["cluster"] = labels
    segmented["max_prob"] = probs.max(axis=1)
    segmented["entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    segmented["log_density"] = gmm.score_samples(X)

    probs_df = pd.DataFrame(probs, columns=[f"p_cluster_{i}" for i in range(k)])
    probs_df["document_id"] = segmented["document_id"].values

    summary = (
        segmented.groupby("cluster", as_index=False)
        .agg(
            n=("document_id", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            mean_saved=("saved_count", "mean"),
            mean_followers=("followers_count", "mean"),
            mean_delivery_cities=("delivery_cities_count", "mean"),
            mean_images=("images_count", "mean"),
            mean_desc_len=("description_len", "mean"),
            pickup_rate=("pickupAvailable", "mean"),
            mean_max_prob=("max_prob", "mean"),
            mean_entropy=("entropy", "mean"),
            top_category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
            top_city=("city_grouped", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
            top_delivery=("deliveryType", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        )
    )

    return gmm, segmented, probs_df, summary


def make_plots(segmented, original_df, X_pca, model_selection, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(original_df["price"].dropna(), bins=20, color="#4C78A8", edgecolor="black")
    axes[0].set_title("Histogram cijena")
    axes[0].set_xlabel("Cijena")
    axes[0].set_ylabel("Frekvencija")

    axes[1].hist(original_df["price_log"].dropna(), bins=20, color="#F58518", edgecolor="black")
    axes[1].set_title("Histogram log-cijena")
    axes[1].set_xlabel("log1p(cijena)")
    axes[1].set_ylabel("Frekvencija")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_histograms.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sub = model_selection[model_selection["feature_set"] == "structured_pca6"]
    for covariance in sorted(sub["covariance"].unique()):
        tmp = sub[sub["covariance"] == covariance].sort_values("k")
        axes[0].plot(tmp["k"], tmp["bic"], marker="o", label=covariance)
        axes[1].plot(tmp["k"], tmp["aic"], marker="o", label=covariance)

    axes[0].set_title("BIC po broju komponenti")
    axes[0].set_xlabel("Broj komponenti k")
    axes[0].set_ylabel("BIC")
    axes[0].legend()

    axes[1].set_title("AIC po broju komponenti")
    axes[1].set_xlabel("Broj komponenti k")
    axes[1].set_ylabel("AIC")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bic_aic.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=segmented["cluster"], alpha=0.8)
    ax.set_title("PCA prostor i GMM segmenti")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Klaster")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=200)
    plt.close(fig)

    plot_df = segmented[["price", "saved_count", "followers_count", "delivery_cities_count"]].copy()
    axes = scatter_matrix(plot_df, figsize=(8, 8), diagonal="hist")
    for ax in axes.flatten():
        ax.tick_params(axis="x", labelrotation=45)
    plt.suptitle("Scatter-matrix odabranih osobina")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_matrix.png"), dpi=200)
    plt.close()

    prob_cols = [col for col in segmented.columns if col.startswith("p_cluster_")]
    arr = segmented[prob_cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(arr, aspect="auto")
    ax.set_title("Heatmap posteriornih vjerovatnoća")
    ax.set_xlabel("Klaster")
    ax.set_ylabel("Proizvod")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "posterior_heatmap.png"), dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = build_analytical_dataset(args.products, args.shops, args.users)
    feature_spaces = build_feature_spaces(df)
    model_selection = evaluate_gmm_grid(feature_spaces)
    model_selection.to_csv(os.path.join(args.out_dir, "model_selection.csv"), index=False)

    X_final = feature_spaces["X_struct_pca"]
    cv_stats = validate_candidate(X_final, k=6, covariance="spherical", n_splits=10)
    stability_stats = bootstrap_stability(X_final, k=6, covariance="spherical", n_boot=10)
    gmm, segmented, probs_df, summary = fit_final_model(
        df,
        X_final,
        k=6,
        covariance="spherical",
    )

    segmented = segmented.merge(probs_df, on="document_id", how="left")
    segmented.to_csv(os.path.join(args.out_dir, "gmm_segments.csv"), index=False)
    probs_df.to_csv(os.path.join(args.out_dir, "gmm_probabilities.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)
    df.to_csv(os.path.join(args.out_dir, "analytical_dataset.csv"), index=False)

    with open(os.path.join(args.out_dir, "final_metrics.txt"), "w", encoding="utf-8") as handle:
        handle.write("FINAL MODEL: structured_pca6 + GMM(spherical, k=6)\n")
        handle.write(f"BIC={gmm.bic(X_final):.6f}\n")
        handle.write(f"AIC={gmm.aic(X_final):.6f}\n")
        handle.write(f"loglik={gmm.score(X_final):.6f}\n")
        handle.write(f"silhouette={silhouette_score(X_final, segmented['cluster']):.6f}\n")
        for key, value in cv_stats.items():
            handle.write(f"{key}={value}\n")
        for key, value in stability_stats.items():
            handle.write(f"{key}={value}\n")

    make_plots(segmented, df, X_final, model_selection, args.out_dir)
    print("Zavrseno. Rezultati su sacuvani u:", args.out_dir)


if __name__ == "__main__":
    main()
