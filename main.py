# main.py — Dataproc / YARN cluster PySpark pipeline for NC Open Policing Project
#
# FLOW:
#  - On first run:
#       * Reads RAW CSV from GCS (NC statewide CSV already extracted from ZIP)
#       * Cleans + enriches data:
#           - time features
#           - labels & flags
#           - ACS county-level demographics from Census ACS for NC counties
#       * Writes enriched Parquet partitioned by year to ENRICHED_PATH
#  - On subsequent runs:
#       * Skips enrichment and directly reads ENRICHED_PATH
#  - Modeling:
#       * Restricts to rows with time_available == 1
#       * For each label in ["citation_issued", "arrest_made"]:
#           - Trains T0 (no time features) and T1 (with time features)
#           - Uses Logistic Regression + optional GBT
#           - Writes models + metrics to GCS
#
# Run (from master VM SSH):
#   spark-submit --master yarn --deploy-mode cluster main.py

import os, sys, math, json
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, FeatureHasher
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# CONFIG / PATHS

BUCKET = "gs://main-bucket-67"

# RAW CSV from Open Policing Project (already extracted from ZIP and uploaded)
# Make sure this exists:
#   gsutil cp nc_statewide_2020_04_01.csv gs://main-bucket-67/raw/nc_statewide_2020_04_01.csv
RAW_URI = f"{BUCKET}/raw/nc_statewide_2020_04_01.csv"

# Enriched Parquet location (directory)
ENRICHED_PATH = f"{BUCKET}/outputs/enriched/nc"

# Outputs base (models + metrics)
OUTPUT_BASE = f"{BUCKET}/outputs"

# Optional Census API key; None = anonymous (OK for small pulls)
CENSUS_API_KEY = "f3f5bf0db3d063859095291d00a0b1027470222a"

# Training knobs
SEED = 67
SAMPLE_FRAC = 1.0 # 0.05, etc. for testing
RUN_GBT = True # also train GBT models
HASH_DEPT = True # hash department_name
NUM_HASH_FEAT = 1 << 18  # 262,144 hashed dims


# SPARK / HDFS HELPERS

def gcs_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))

def write_json_to_gcs(spark, obj, path: str):
    (spark.createDataFrame([json.dumps(obj)], "string")
          .toDF("json")
          .coalesce(1)
          .write.mode("overwrite").text(path))


# CLEANING / FEATURE UTILS

YN_COLS = ["citation_issued", "arrest_made", "warning_issued", "search_conducted", "frisk_performed"]

def clean_column_names(df):
    return df.select([F.col(c).alias(c.strip().lower().replace(" ", "_")) for c in df.columns])

def normalize_booleans(df):
    for c in [x for x in YN_COLS if x in df.columns]:
        df = df.withColumn(
            c,
            F.when(F.col(c).rlike("^(?i)1|y|yes|true$"), 1).otherwise(0).cast("int")
        )
    return df

def standardize_strings(df, cols):
    for c in cols:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.trim(F.lower(F.col(c))).isin("nan", ""), None)
                 .otherwise(F.trim(F.col(c)))
            )
    return df

def clip_age(df):
    if "subject_age" in df.columns:
        df = df.withColumn(
            "subject_age",
            F.when((F.col("subject_age") < 16) | (F.col("subject_age") > 100), None)
             .otherwise(F.col("subject_age").cast("double"))
        )
    return df

def make_time_features(df):
    df = df.withColumn("date", F.to_date("date"))
    df = df.withColumn(
        "stop_ts",
        F.to_timestamp(F.concat_ws(" ", F.col("date"), F.col("time")), "yyyy-MM-dd HH:mm:ss")
    )
    df = df.withColumn("year", F.year("date"))
    df = df.withColumn("time_available", F.col("time").isNotNull().cast("int"))

    # Hour-of-day and cyclic encodings (only where time is available)
    df = df.withColumn(
        "hour_of_day",
        F.when(F.col("time_available") == 1, F.hour("stop_ts"))
    )
    df = df.withColumn(
        "hour_sin",
        F.when(
            F.col("time_available") == 1,
            F.sin(2 * math.pi * F.col("hour_of_day") / F.lit(24.0))
        )
    )
    df = df.withColumn(
        "hour_cos",
        F.when(
            F.col("time_available") == 1,
            F.cos(2 * math.pi * F.col("hour_of_day") / F.lit(24.0))
        )
    )

    # Use built-in dayofweek: 1 = Sunday, 7 = Saturday
    df = df.withColumn(
        "day_of_week",
        F.when(F.col("time_available") == 1, F.dayofweek("stop_ts"))
    )

    # Weekend = Saturday (7) or Sunday (1)
    df = df.withColumn(
        "is_weekend",
        F.when(
            (F.col("time_available") == 1) & (F.col("day_of_week").isin(1, 7)),
            F.lit(1)
        ).when(F.col("time_available") == 1, F.lit(0))
    )

    return df


def add_basic_labels_flags(df):
    if "citation_issued" in df.columns:
        df = df.withColumn("citation_yn", F.col("citation_issued"))
    if "arrest_made" in df.columns:
        df = df.withColumn("arrest_yn", F.col("arrest_made"))
    if "outcome" in df.columns:
        df = df.withColumn(
            "outcome_multiclass",
            F.when(F.col("arrest_made") == 1, "arrest")
             .when(F.col("citation_issued") == 1, "citation")
             .when(F.col("warning_issued") == 1, "warning")
             .otherwise(F.lit("other"))
        )
    if "location" in df.columns:
        df = df.withColumn("location_known", F.col("location").isNotNull().cast("int"))
    if "county_name" in df.columns:
        df = df.withColumn("county_known", F.col("county_name").isNotNull().cast("int"))

    for c in ["subject_race", "subject_sex"]:
        if c in df.columns:
            df = df.withColumn(c, F.lower(F.col(c)))

    # normalized county name for ACS join
    if "county_name" in df.columns:
        df = df.withColumn(
            "county_name_norm",
            F.lower(
                F.regexp_replace(
                    F.regexp_replace(F.col("county_name"), r"\s*County$", ""),
                    r",$", ""
                )
            )
        )
    return df


# ACS FETCH AND JOIN

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def census_get(year: int, dataset_suffix: str, vars_list):
    base = f"https://api.census.gov/data/{year}/acs/acs5{dataset_suffix}"
    params = {
        "get": ",".join(["NAME"] + vars_list),
        "for": "county:*",
        "in": "state:37",  # 37 = North Carolina
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    url = base + "?" + urlencode(params)
    try:
        with urlopen(url) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (HTTPError, URLError) as e:
        print(f"[WARN] ACS fetch failed for {year} {dataset_suffix}: {e}")
        return []
    if not data or len(data) < 2:
        return []
    header = data[0]
    rows = data[1:]
    out = []
    for row in rows:
        obj = dict(zip(header, row))
        obj["_year"] = year
        out.append(obj)
    return out

def fetch_nc_acs(years):
    # variables:
    #  median income: B19013_001E  (acs5)
    #  poverty %:     S1701_C03_001E (subject)
    #  unemployment%: S2301_C04_001E (subject)
    #  bachelor's+:   DP02_0068PE (profile)
    acs_rows = []
    for y in sorted(set(years)):
        acs_rows += census_get(y, "", ["B19013_001E"])
        acs_rows += census_get(y, "/subject", ["S1701_C03_001E", "S2301_C04_001E"])
        acs_rows += census_get(y, "/profile", ["DP02_0068PE"])

    by_key = {}
    for r in acs_rows:
        key = (r.get("_year"), r.get("state"), r.get("county"))
        if key not in by_key:
            by_key[key] = {"_year": key[0], "state": key[1], "county": key[2]}
        by_key[key].update(r)

    result = []
    for k, r in by_key.items():
        name = r.get("NAME", "")
        county_norm = name.lower().replace(" county, north carolina", "").strip()
        result.append({
            "acs_year": r["_year"],
            "state": r.get("state"),
            "county_fips3": r.get("county"),
            "county_name_norm": county_norm,
            "median_income": _to_float(r.get("B19013_001E")),
            "poverty_rate": _to_float(r.get("S1701_C03_001E")),
            "unemployment_rate": _to_float(r.get("S2301_C04_001E")),
            "edu_bachelor_pct": _to_float(r.get("DP02_0068PE")),
        })
    return result

def add_acs_join(df, spark):
    if "county_name_norm" not in df.columns or "year" not in df.columns:
        return df

    years_present = [r["year"] for r in df.select("year").distinct().collect() if r["year"] is not None]
    if not years_present:
        return df

    # Map early years to first ACS 5-year we know (e.g., 2009)
    acs_years = []
    for y in years_present:
        if y is None:
            continue
        if y < 2009:
            acs_years.append(2009)
        else:
            acs_years.append(y)
    acs_years = sorted(set(acs_years))
    print(f"[INFO] Fetching ACS for years: {acs_years}")
    rows = fetch_nc_acs(acs_years)
    if not rows:
        print("[WARN] No ACS rows; continuing without ACS enrichment.")
        return df

    acs_df = spark.createDataFrame(rows)
    df = df.withColumn("acs_year",
                       F.when(F.col("year") < 2009, F.lit(2009)).otherwise(F.col("year")))
    df = df.join(
        F.broadcast(acs_df),
        on=[df["county_name_norm"] == acs_df["county_name_norm"],
            df["acs_year"] == acs_df["acs_year"]],
        how="left"
    ).drop(acs_df["county_name_norm"]).drop(acs_df["acs_year"])
    return df


# MODEL PIPELINES

def build_pipelines(train_df, include_time: bool, label_col: str, use_hash: bool, num_hash_features: int):
    cat_small = [c for c in ["subject_race", "subject_sex", "reason_for_stop", "type", "county_name"]
                 if c in train_df.columns]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_small]
    enc = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_small],
        outputCols=[f"{c}_oh" for c in cat_small]
    )

    stages = indexers + [enc]
    inputs = [f"{c}_oh" for c in cat_small]

    if use_hash and "department_name" in train_df.columns:
        hasher = FeatureHasher(inputCols=["department_name"], outputCol="dept_hash",
                               numFeatures=num_hash_features)
        stages += [hasher]
        inputs += ["dept_hash"]

    num_cols = []
    for c in ["subject_age", "median_income", "poverty_rate", "unemployment_rate",
              "edu_bachelor_pct", "dept_freq", "location_known", "county_known"]:
        if c in train_df.columns:
            num_cols.append(c)

    if include_time:
        for c in ["time_available", "hour_of_day", "hour_sin", "hour_cos", "is_weekend"]:
            if c in train_df.columns:
                num_cols.append(c)

    assembler = VectorAssembler(inputCols=inputs + num_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    lr = LogisticRegression(labelCol=label_col, featuresCol="features",
                            maxIter=50, regParam=0.01, elasticNetParam=0.1, weightCol="class_w")
    gbt = GBTClassifier(labelCol=label_col, featuresCol="features",
                        maxDepth=6, maxIter=60, stepSize=0.1, subsamplingRate=0.8)

    pipe_lr  = Pipeline(stages=stages + [assembler, scaler, lr])
    pipe_gbt = Pipeline(stages=stages + [assembler, scaler, gbt])

    return pipe_lr, pipe_gbt

def fairness_by(df, label_col, pred_col, group_col):
    g = df.groupBy(group_col).agg(
        F.sum(F.when((F.col(label_col) == 1) & (F.col(pred_col) == 1), 1).otherwise(0)).alias("TP"),
        F.sum(F.when(F.col(label_col) == 1, 1).otherwise(0)).alias("P"),
        F.sum(F.when((F.col(label_col) == 0) & (F.col(pred_col) == 1), 1).otherwise(0)).alias("FP"),
        F.sum(F.when(F.col(label_col) == 0, 1).otherwise(0)).alias("N"),
        F.sum(F.when(F.col(pred_col) == 1, 1).otherwise(0)).alias("Pos")
    )
    return (g.withColumn("TPR", F.when(F.col("P") > 0, F.col("TP") / F.col("P")))
             .withColumn("FPR", F.when(F.col("N") > 0, F.col("FP") / F.col("N")))
             .withColumn("Rate", F.when((F.col("P") + F.col("N")) > 0,
                                        F.col("Pos") / (F.col("P") + F.col("N")))))


# MAIN

if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("nc-opp-enrich-train")
             .config("spark.sql.shuffle.partitions", "400")
             .getOrCreate())

    def build_enriched_from_raw(spark: SparkSession):
        print(f"[INFO] Building enriched dataset from raw CSV: {RAW_URI}")

        # Directly read CSV from GCS (no ZIP or subprocess)
        raw = spark.read.option("header", True).csv(RAW_URI)

        df = clean_column_names(raw)
        df = standardize_strings(df, ["location", "county_name", "subject_race", "subject_sex",
                                      "officer_id_hash", "department_name", "type",
                                      "outcome", "reason_for_stop"])
        df = normalize_booleans(df)
        df = clip_age(df)
        df = make_time_features(df)
        df = add_basic_labels_flags(df)

        # Global dept_freq (we'll refine with train-only later)
        if "department_name" in df.columns:
            dept_freq_all = (df.groupBy("department_name").count()
                               .withColumnRenamed("count", "dept_freq"))
            df = df.join(F.broadcast(dept_freq_all), on="department_name", how="left")

        # ACS join
        if "county_name_norm" in df.columns:
            df = add_acs_join(df, spark)

        # Save enriched dataset
        (df.write
           .mode("overwrite")
           .partitionBy("year")
           .parquet(ENRICHED_PATH))
        print(f"[INFO] Enriched parquet written to {ENRICHED_PATH}")
        return df

    # Enrichment: build or reuse
    if gcs_exists(spark, ENRICHED_PATH):
        print(f"[INFO] Enriched dataset exists at {ENRICHED_PATH} — attempting to load.")
        try:
            enriched_df = spark.read.parquet(ENRICHED_PATH)
        except Exception as e:
            print(f"[WARN] Failed to read existing enriched dataset: {e}")
            print("[WARN] Rebuilding enriched dataset from raw CSV.")
            enriched_df = build_enriched_from_raw(spark)
    else:
        print(f"[INFO] Enriched dataset NOT found; building from raw CSV: {RAW_URI}")
        enriched_df = build_enriched_from_raw(spark)

    # Optional sampling
    if SAMPLE_FRAC < 1.0:
        enriched_df = enriched_df.sample(False, SAMPLE_FRAC, seed=SEED)

    # Restrict to time-present cohort for T0 vs T1 comparison
    df_time = enriched_df.filter(F.col("time_available") == 1)

    # Two label tasks
    TASKS = ["citation_issued", "arrest_made"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for label_col in TASKS:
        print(f"[INFO] Training label: {label_col}")

        base_df = df_time.filter(F.col(label_col).isNotNull())

        # 80/20 random split
        train, test = base_df.randomSplit([0.8, 0.2], seed=SEED)

        # Class weights
        pos = train.filter(F.col(label_col) == 1).count()
        neg = train.filter(F.col(label_col) == 0).count()
        N   = max(pos + neg, 1)
        wpos = N / (2.0 * max(pos, 1))
        wneg = N / (2.0 * max(neg, 1))
        train = train.withColumn("class_w", F.when(F.col(label_col) == 1, F.lit(wpos)).otherwise(F.lit(wneg)))
        test  =  test.withColumn("class_w", F.when(F.col(label_col) == 1, F.lit(wpos)).otherwise(F.lit(wneg)))

        # Train-only dept_freq (override global, avoids leakage drift)
        if not HASH_DEPT and "department_name" in train.columns:
            tr_freq = (train.groupBy("department_name").count()
                       .withColumnRenamed("count", "dept_freq"))
            train = train.join(F.broadcast(tr_freq), on="department_name", how="left")
            test  = test.join(F.broadcast(tr_freq),  on="department_name", how="left")

        # Pipelines: T0 (no time) and T1 (with time)
        pipe_lr_T0,  pipe_gbt_T0  = build_pipelines(train, include_time=False, label_col=label_col,
                                                    use_hash=HASH_DEPT, num_hash_features=NUM_HASH_FEAT)
        pipe_lr_T1,  pipe_gbt_T1  = build_pipelines(train, include_time=True,  label_col=label_col,
                                                    use_hash=HASH_DEPT, num_hash_features=NUM_HASH_FEAT)

        # Fit
        m_lr_T0 = pipe_lr_T0.fit(train)
        m_lr_T1 = pipe_lr_T1.fit(train)
        if RUN_GBT:
            m_gbt_T0 = pipe_gbt_T0.fit(train)
            m_gbt_T1 = pipe_gbt_T1.fit(train)
        else:
            m_gbt_T0 = m_gbt_T1 = None

        # Evaluate
        eval_roc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
        eval_pr  = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR")

        def eval_model(model, name):
            if not model:
                return {}
            pred = (model.transform(test)
                    .select(label_col, "prediction", "probability", "subject_race", "subject_sex"))
            auroc = eval_roc.evaluate(pred)
            aupr  = eval_pr.evaluate(pred)
            fair_race = fairness_by(pred, label_col, "prediction", "subject_race").toPandas().to_dict(orient="list")
            fair_sex  = fairness_by(pred, label_col, "prediction", "subject_sex").toPandas().to_dict(orient="list")
            return {"model": name, "auroc": auroc, "aupr": aupr,
                    "fair_race": fair_race, "fair_sex": fair_sex}

        results = []
        results.append(eval_model(m_lr_T0,  f"LR_T0_no_time_{label_col}"))
        results.append(eval_model(m_lr_T1,  f"LR_T1_with_time_{label_col}"))
        if m_gbt_T0:
            results.append(eval_model(m_gbt_T0, f"GBT_T0_no_time_{label_col}"))
        if m_gbt_T1:
            results.append(eval_model(m_gbt_T1, f"GBT_T1_with_time_{label_col}"))

        # Persist artifacts
        base_out = f"{OUTPUT_BASE}/{label_col}/{ts}"
        m_lr_T0.write().overwrite().save(f"{base_out}/models/LR_T0")
        m_lr_T1.write().overwrite().save(f"{base_out}/models/LR_T1")
        if m_gbt_T0:
            m_gbt_T0.write().overwrite().save(f"{base_out}/models/GBT_T0")
        if m_gbt_T1:
            m_gbt_T1.write().overwrite().save(f"{base_out}/models/GBT_T1")

        write_json_to_gcs(
            spark,
            {"results": [r for r in results if r],
             "class_counts_train": {"pos": pos, "neg": neg, "N": pos + neg},
             "weights": {"w_pos": wpos, "w_neg": wneg}},
            f"{base_out}/metrics/json"
        )

        print(f"[INFO] DONE label={label_col}. Artifacts at: {base_out}")

    spark.stop()
