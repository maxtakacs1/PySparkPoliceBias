# main.py — Dataproc-ready PySpark pipeline for NC OPP
# - Enrichment step:
#   * If enriched Parquet exists in GCS -> skip enrichment
#   * Else read your single ZIP from GCS, extract CSV on driver, upload CSV to GCS,
#     clean + add time features + add ACS county demographics (NC) + save Parquet
# - Modeling:
#   * Train on time-present cohort only; compare T0 (no time features) vs T1 (with time)
#   * Algorithms: Logistic Regression (baseline) and optional GBT
# - No CLI args for paths/seed/year-split — set constants below.

import os, sys, math, json, zipfile, subprocess
from datetime import datetime
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, FeatureHasher
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# =============================================================================
# 0) CONFIG — EDIT THESE TWO LINES BEFORE RUNNING
# =============================================================================
BUCKET = "gs://YOUR_BUCKET"  # <-- CHANGE THIS to your bucket
RAW_URI = f"{BUCKET}/raw/yg821jf8611_nc_statewide_2020_04_01.csv.zip"

# Optional: Census API key (None = anonymous; often works for small pulls)
CENSUS_API_KEY = None  # or "YOUR_CENSUS_KEY"

# Output and enrichment paths
ENRICHED_PATH = f"{BUCKET}/outputs/enriched/nc"       # Parquet folder to write/reuse
OUTPUT_BASE   = f"{BUCKET}/outputs"                   # Where metrics/models go

# Training knobs (no CLI args)
SEED          = 42
SAMPLE_FRAC   = 1.0         # e.g., 0.05 for warm-up
RUN_GBT       = True        # also train Gradient-Boosted Trees
HASH_DEPT     = True        # hash high-cardinality department_name
NUM_HASH_FEAT = 1 << 18     # 262k hashed dims

# =============================================================================
# 1) SPARK HELPERS
# =============================================================================
def gcs_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))

def write_json_to_gcs(spark, obj, path):
    (spark.createDataFrame([json.dumps(obj)], "string")
          .toDF("json")
          .coalesce(1)
          .write.mode("overwrite").text(path))

def run(cmd: str):
    print(f"[CMD] {cmd}")
    subprocess.check_call(cmd, shell=True)

# =============================================================================
# 2) ZIP → CSV (GCS)
# =============================================================================
def ensure_csv_on_gcs_from_zip(raw_zip_uri: str, target_csv_uri: str):
    os.makedirs("/tmp/nc_raw", exist_ok=True)
    local_zip = "/tmp/nc_raw/data.zip"
    local_csv = "/tmp/nc_raw/nc_extracted.csv"

    run(f"gsutil cp {raw_zip_uri} {local_zip}")

    with zipfile.ZipFile(local_zip, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("ZIP contains no CSV files.")
        first = csv_members[0]
        with zf.open(first) as f_in, open(local_csv, "wb") as f_out:
            f_out.write(f_in.read())

    run(f"gsutil cp {local_csv} {target_csv_uri}")

# =============================================================================
# 3) CLEANING / FEATURE UTILS
# =============================================================================
YN_COLS = ["citation_issued","arrest_made","warning_issued","search_conducted","frisk_performed"]

def clean_column_names(df):
    return df.select([F.col(c).alias(c.strip().lower().replace(" ","_")) for c in df.columns])

def normalize_booleans(df):
    for c in [x for x in YN_COLS if x in df.columns]:
        df = df.withColumn(c, F.when(F.col(c).rlike("^(?i)1|y|yes|true$"), 1).otherwise(0).cast("int"))
    return df

def standardize_strings(df, cols):
    for c in cols:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.trim(F.lower(F.col(c))).isin("nan",""), None).otherwise(F.trim(F.col(c)))
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
    df = df.withColumn("stop_ts", F.to_timestamp(F.concat_ws(" ", F.col("date"), F.col("time")), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("year", F.year("date"))
    df = df.withColumn("time_available", F.col("time").isNotNull().cast("int"))
    df = df.withColumn("hour_of_day", F.when(F.col("time_available")==1, F.hour("stop_ts")))
    df = df.withColumn("hour_sin", F.when(F.col("time_available")==1, F.sin(2*math.pi * F.col("hour_of_day")/24.0)))
    df = df.withColumn("hour_cos", F.when(F.col("time_available")==1, F.cos(2*math.pi * F.col("hour_of_day")/24.0)))
    df = df.withColumn("day_of_week", F.when(F.col("time_available")==1, F.date_format("stop_ts", "u").cast("int")))
    df = df.withColumn("is_weekend", F.when(F.col("time_available")==1, (F.col("day_of_week")>=6).cast("int")))
    return df

def add_basic_labels_flags(df):
    if "citation_issued" in df.columns:
        df = df.withColumn("citation_yn", F.col("citation_issued"))
    if "arrest_made" in df.columns:
        df = df.withColumn("arrest_yn", F.col("arrest_made"))
    if "outcome" in df.columns:
        df = df.withColumn(
            "outcome_multiclass",
            F.when(F.col("arrest_made")==1, "arrest")
             .when(F.col("citation_issued")==1, "citation")
             .when(F.col("warning_issued")==1, "warning")
             .otherwise(F.lit("other"))
        )
    if "location" in df.columns:
        df = df.withColumn("location_known", F.col("location").isNotNull().cast("int"))
    if "county_name" in df.columns:
        df = df.withColumn("county_known", F.col("county_name").isNotNull().cast("int"))
    for c in ["subject_race","subject_sex"]:
        if c in df.columns:
            df = df.withColumn(c, F.lower(F.col(c)))
    # normalized county name for joining (remove " County"/commas, lower)
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

# =============================================================================
# 4) ACS FETCH (NC counties only)
# =============================================================================
def census_get(year: int, dataset_suffix: str, vars_list):
    base = f"https://api.census.gov/data/{year}/acs/acs5{dataset_suffix}"
    params = {
        "get": ",".join(["NAME"] + vars_list),
        "for": "county:*",
        "in": "state:37",  # 37 = NC
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
    # Variables by dataset:
    # acs/acs5:       B19013_001E -> median income
    # acs/acs5/subject: S1701_C03_001E -> poverty %, S2301_C04_001E -> unemployment %
    # acs/acs5/profile: DP02_0068PE -> % bachelor's or higher
    acs_rows = []
    for y in sorted(set(years)):
        # Some early years may not exist; try and skip gracefully
        acs_rows += census_get(y, "", ["B19013_001E"])  # median income
        acs_rows += census_get(y, "/subject", ["S1701_C03_001E","S2301_C04_001E"])  # poverty %, unemployment %
        acs_rows += census_get(y, "/profile", ["DP02_0068PE"])  # bachelor %
    # Consolidate by (year, county)
    by_key = {}
    for r in acs_rows:
        key = (r.get("_year"), r.get("state"), r.get("county"))
        if key not in by_key: by_key[key] = {"_year": key[0], "state": key[1], "county": key[2]}
        by_key[key].update(r)
    # Convert to Spark rows with normalized county name
    result = []
    for k, r in by_key.items():
        name = r.get("NAME", "")  # e.g., "Wake County, North Carolina"
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

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def add_acs_join(df, spark):
    # Determine which years to fetch (ACS 5-year widely available >=2009; clamp)
    years_present = [r["year"] for r in df.select("year").distinct().collect() if r["year"] is not None]
    if not years_present:
        return df
    acs_years = []
    for y in years_present:
        if y is None: continue
        if y < 2009: acs_years.append(2009)
        else: acs_years.append(y)
    acs_years = sorted(set(acs_years))
    print(f"[INFO] Fetching ACS for years: {acs_years}")
    rows = fetch_nc_acs(acs_years)
    if not rows:
        print("[WARN] ACS fetch returned no rows; continuing without ACS enrichment.")
        return df
    acs_df = spark.createDataFrame(rows)
    # Join key: (county_name_norm, acs_year == year or clamped year for pre-2009)
    df = df.withColumn("acs_year", F.when(F.col("year") < 2009, F.lit(2009)).otherwise(F.col("year")))
    df = df.join(F.broadcast(acs_df),
                 on=[df["county_name_norm"] == acs_df["county_name_norm"], df["acs_year"] == acs_df["acs_year"]],
                 how="left") \
           .drop(acs_df["county_name_norm"]).drop(acs_df["acs_year"])
    return df

# =============================================================================
# 5) MODEL PIPELINES
# =============================================================================
def build_pipelines(train_df, include_time: bool, label_col: str, use_hash: bool, num_hash_features: int):
    cat_small = [c for c in ["subject_race","subject_sex","reason_for_stop","type","county_name"] if c in train_df.columns]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_small]
    enc = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_small],
        outputCols=[f"{c}_oh" for c in cat_small]
    )
    stages = indexers + [enc]
    inputs = [f"{c}_oh" for c in cat_small]

    if use_hash and "department_name" in train_df.columns:
        hasher = FeatureHasher(inputCols=["department_name"], outputCol="dept_hash", numFeatures=num_hash_features)
        stages += [hasher]
        inputs += ["dept_hash"]
    else:
        hasher = None
        # fallback frequency (computed on train only)
        if "department_name" in train_df.columns and "dept_freq" in train_df.columns:
            pass  # num column added below

    num_cols = []
    for c in ["subject_age","median_income","poverty_rate","unemployment_rate","edu_bachelor_pct",
              "dept_freq","location_known","county_known"]:
        if c in train_df.columns: num_cols.append(c)

    if include_time:
        for c in ["time_available","hour_of_day","hour_sin","hour_cos","is_weekend"]:
            if c in train_df.columns: num_cols.append(c)

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
        F.sum(F.when((F.col(label_col)==1) & (F.col(pred_col)==1), 1).otherwise(0)).alias("TP"),
        F.sum(F.when(F.col(label_col)==1, 1).otherwise(0)).alias("P"),
        F.sum(F.when((F.col(label_col)==0) & (F.col(pred_col)==1), 1).otherwise(0)).alias("FP"),
        F.sum(F.when(F.col(label_col)==0, 1).otherwise(0)).alias("N"),
        F.sum(F.when(F.col(pred_col)==1, 1).otherwise(0)).alias("Pos")
    )
    return (g.withColumn("TPR", F.when(F.col("P")>0, F.col("TP")/F.col("P")))
             .withColumn("FPR", F.when(F.col("N")>0, F.col("FP")/F.col("N")))
             .withColumn("Rate", F.when((F.col("P")+F.col("N"))>0, F.col("Pos")/(F.col("P")+F.col("N"))))
            )

# =============================================================================
# 6) MAIN
# =============================================================================
if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("nc-opp-enrich-train")
             .config("spark.sql.shuffle.partitions", "400")
             .getOrCreate())

    # ---------------- Enrichment (idempotent) ----------------
    if gcs_exists(spark, ENRICHED_PATH):
        print(f"[INFO] Enriched dataset exists at {ENRICHED_PATH} — loading.")
        enriched_df = spark.read.parquet(ENRICHED_PATH)
    else:
        print(f"[INFO] Building enrichment from raw ZIP: {RAW_URI}")
        # Derive a CSV path next to the ZIP
        parts = RAW_URI.replace("gs://","").split("/",1)
        bucket = parts[0]; key = parts[1] if len(parts)>1 else ""
        base_prefix = "/".join(key.split("/")[:-1])
        csv_uri = f"gs://{bucket}/{base_prefix}/_extracted_nc.csv" if base_prefix else f"gs://{bucket}/_extracted_nc.csv"

        ensure_csv_on_gcs_from_zip(RAW_URI, csv_uri)

        raw = spark.read.option("header", True).csv(csv_uri)
        df  = clean_column_names(raw)
        df  = standardize_strings(df, ["location","county_name","subject_race","subject_sex",
                                       "officer_id_hash","department_name","type","outcome","reason_for_stop"])
        df  = normalize_booleans(df)
        df  = clip_age(df)
        df  = make_time_features(df)
        df  = add_basic_labels_flags(df)

        # department_name frequency (global seed, later replaced by train-only mapping)
        if "department_name" in df.columns:
            dept_freq_all = df.groupBy("department_name").count().withColumnRenamed("count","dept_freq")
            df = df.join(F.broadcast(dept_freq_all), on="department_name", how="left")

        # --- ACS join (NC counties) ---
        if "county_name_norm" in df.columns:
            df = add_acs_join(df, spark)

        # Write enriched parquet partitioned by year
        (df.write.mode("overwrite").partitionBy("year").parquet(ENRICHED_PATH))
        print(f"[INFO] Enriched parquet written to {ENRICHED_PATH}")
        enriched_df = df

    # Optional sampling
    if SAMPLE_FRAC < 1.0:
        enriched_df = enriched_df.sample(False, SAMPLE_FRAC, seed=SEED)

    # Build time-present cohort for apples-to-apples comparison
    # (Train/test uses ONLY rows with parsed 'time')
    df_all  = enriched_df
    df_time = df_all.filter(F.col("time_available")==1)

    # Two tasks to run back-to-back
    TASKS = ["citation_issued","arrest_made"]

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    for label_col in TASKS:
        # Filter rows with label present
        base_df = df_time.filter(F.col(label_col).isNotNull())

        # Train/test split (80/20)
        train, test = base_df.randomSplit([0.8, 0.2], seed=SEED)

        # Class weights (on train)
        pos = train.filter(F.col(label_col)==1).count()
        neg = train.filter(F.col(label_col)==0).count()
        N   = max(pos+neg, 1)
        wpos, wneg = N/(2.0*max(pos,1)), N/(2.0*max(neg,1))
        train = train.withColumn("class_w", F.when(F.col(label_col)==1, F.lit(wpos)).otherwise(F.lit(wneg)))
        test  =  test.withColumn("class_w", F.when(F.col(label_col)==1, F.lit(wpos)).otherwise(F.lit(wneg)))

        # Train-only dept freq mapping to avoid leakage drift
        if not HASH_DEPT and "department_name" in train.columns:
            tr_freq = train.groupBy("department_name").count().withColumnRenamed("count","dept_freq")
            train = train.join(F.broadcast(tr_freq), on="department_name", how="left")
            test  =  test.join(F.broadcast(tr_freq),  on="department_name", how="left")

        # Pipelines
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
            if not model: return {}
            pred = (model.transform(test)
                    .select(label_col, "prediction", "probability", "subject_race", "subject_sex"))
            auroc = eval_roc.evaluate(pred)
            aupr  = eval_pr.evaluate(pred)
            fair_race = fairness_by(pred, label_col, "prediction", "subject_race").toPandas().to_dict(orient="list")
            fair_sex  = fairness_by(pred, label_col, "prediction", "subject_sex").toPandas().to_dict(orient="list")
            return {"model": name, "auroc": auroc, "aupr": aupr, "fair_race": fair_race, "fair_sex": fair_sex}

        results = []
        results.append(eval_model(m_lr_T0,  f"LR_T0_no_time_{label_col}"))
        results.append(eval_model(m_lr_T1,  f"LR_T1_with_time_{label_col}"))
        if m_gbt_T0: results.append(eval_model(m_gbt_T0, f"GBT_T0_no_time_{label_col}"))
        if m_gbt_T1: results.append(eval_model(m_gbt_T1, f"GBT_T1_with_time_{label_col}"))

        # Persist artifacts
        base = f"{OUTPUT_BASE}/{label_col}/{ts}"
        # Save models
        m_lr_T0.write().overwrite().save(f"{base}/models/LR_T0")
        m_lr_T1.write().overwrite().save(f"{base}/models/LR_T1")
        if m_gbt_T0: m_gbt_T0.write().overwrite().save(f"{base}/models/GBT_T0")
        if m_gbt_T1: m_gbt_T1.write().overwrite().save(f"{base}/models/GBT_T1")
        # Save metrics
        write_json_to_gcs(spark,
                          {"results": [r for r in results if r],
                           "class_counts_train": {"pos": pos, "neg": neg, "N": pos+neg},
                           "weights": {"w_pos": wpos, "w_neg": wneg}},
                          f"{base}/metrics/json")

        print(f"[INFO] DONE {label_col}. Artifacts at: {base}")

    spark.stop()
