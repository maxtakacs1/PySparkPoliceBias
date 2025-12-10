# NC Open Policing – Spark ML Project on GCP Dataproc

This project uses **PySpark on Google Cloud Dataproc** to analyze the **North Carolina Statewide Open Policing** data and:

* Enrich the data with:

  * Time-of-day / day-of-week features
  * County-level socio-economic features from **ACS**
  * Department frequency features
* Train two prediction tasks:

  * `citation_issued`
  * `arrest_made`
* For each task, compare:

  * **T0** – model **without time features**
  * **T1** – model **with time features**
* Compare performance across three **Dataproc cluster configurations**:

  1. Single node: 4 cores, 16 GB
  2. 1 master + 2 workers: 4 cores, 16 GB per node
  3. 1 master + 4 workers: 4 cores, 16 GB per node

The workflow is:

> Create Dataproc cluster → SSH into master → sanity-check `pyspark` → upload `main.py` → run
> `spark-submit --master yarn --deploy-mode cluster main.py`

All project code is in `main.py` and is written in **PySpark**.

---

## 1. Project Files & Config

### `main.py`

* Main Dataproc / YARN driver script.
* Handles:

  * Reading **raw CSV** from GCS.
  * Enrichment (time features, ACS join, dept frequency).
  * Writing and reusing **enriched Parquet**.
  * Training and evaluating models.

At the top of `main.py` you’ll see:

```python
BUCKET = "gs://main-bucket-67"

# RAW CSV from Open Policing Project (already extracted from ZIP and uploaded)
RAW_URI = f"{BUCKET}/raw/nc_statewide_2020_04_01.csv"

ENRICHED_PATH = f"{BUCKET}/outputs/enriched/nc"
OUTPUT_BASE   = f"{BUCKET}/outputs"

CENSUS_API_KEY = "..."  # your ACS API key

SEED        = 67
SAMPLE_FRAC = 0.50      # fraction of enriched data used for modeling
RUN_GBT     = True      # also train GBT models
```

You only need to change:

* `BUCKET` if your bucket name is different.
* You can optionally tweak `SAMPLE_FRAC`, `RUN_GBT`, etc.

---

## 2. Data Preparation in GCS

### 2.1. Create / choose GCS bucket

In the GCP Console:

1. **Navigation ☰ → Cloud Storage → Buckets → Create**
2. Create a bucket (e.g. `main-bucket-67`) in the same region you’ll use for Dataproc (e.g. `us-central1`).

In this README we assume:

```text
gs://main-bucket-67/
```

### 2.2. Upload the NC statewide CSV

You start from the **ZIP** from the Stanford Open Policing Project.

On your local machine or on the Dataproc master VM:

```bash
# Example on the Dataproc master VM or Cloud Shell

# 1) Copy ZIP to current directory (if it’s already in GCS)
gsutil cp gs://main-bucket-67/raw/yg821jf8611_nc_statewide_2020_04_01.csv.zip .

# 2) Unzip
unzip yg821jf8611_nc_statewide_2020_04_01.csv.zip

# 3) Upload CSV back to GCS
gsutil cp yg821jf8611_nc_statewide_2020_04_01.csv \
  gs://main-bucket-67/raw/nc_statewide_2020_04_01.csv
```

Your bucket layout will look like:

```text
gs://main-bucket-67/
  raw/
    yg821jf8611_nc_statewide_2020_04_01.csv.zip   # original
    nc_statewide_2020_04_01.csv                   # raw CSV used by main.py
  outputs/
    enriched/   # enriched data under 'nc'
    ...         # model outputs per label
```

Make sure `RAW_URI` in `main.py` points to this CSV.

---

## 3. What `main.py` Does

### 3.1. Enrichment (build or reuse)

At runtime, `main.py` does:

1. **Check if enriched data exists**:

   ```python
   if gcs_exists(spark, ENRICHED_PATH):
       try:
           enriched_df = spark.read.parquet(ENRICHED_PATH)
       except Exception:
           # If read fails (e.g., corrupted/empty dataset), rebuild from raw
           enriched_df = build_enriched_from_raw(spark)
   else:
       enriched_df = build_enriched_from_raw(spark)
   ```

2. **`build_enriched_from_raw`** does:

   * Reads **raw CSV** from `RAW_URI` (GCS).
   * Cleans & standardizes:

     * Column names → lowercase with `_`
     * String columns → strips `"nan"` / empty → `NULL`
     * Boolean-like columns → 0/1 (`citation_issued`, `arrest_made`, etc.)
     * `subject_age` clipped to [16, 100]
   * Adds **time features**:

     * `time_available`, `hour_of_day`, `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `year`
   * Adds **labels and flags**:

     * `citation_yn`, `arrest_yn`
     * `outcome_multiclass` (arrest / citation / warning / other)
     * `location_known`, `county_known`
     * `county_name_norm` (for ACS join)
   * Computes global **department frequency**:

     * `dept_freq` = count of rows per `department_name`
   * Calls **ACS API** (via `CENSUS_API_KEY`) for NC counties:

     * `median_income`
     * `poverty_rate`
     * `unemployment_rate`
     * `edu_bachelor_pct`
   * Joins ACS data onto stops based on normalized county name and year (or nearest ACS year).
   * Writes **partitioned Parquet** to:

     ```text
     gs://main-bucket-67/outputs/enriched/nc/
     ```

3. On **subsequent runs**, if the Parquet is valid, it just loads from `ENRICHED_PATH` without calling ACS or recomputing features.

4. After enrichment, if `SAMPLE_FRAC < 1.0`, the code downsamples for training:

   ```python
   enriched_df = enriched_df.sample(False, SAMPLE_FRAC, seed=SEED)
   ```

### 3.2. Modeling

For each label in `["citation_issued", "arrest_made"]`:

1. **Filter** to rows with `time_available == 1` and non-null label.

2. Do an **80/20 random split** (`train`, `test`) with seed.

3. Compute **class weights**:

   * Higher weight for the minority class (label = 1), stored in `class_w`.
   * Used as `weightCol="class_w"` in Logistic Regression.

4. Build two feature pipelines:

   * **T0 (no time)**: demographics, dept info, county + ACS, but **no** time features.
   * **T1 (with time)**: T0 features + `time_available`, `hour_of_day`, `hour_sin`, `hour_cos`, `is_weekend`.

   Both pipelines:

   * Treat small categoricals (`subject_race`, `subject_sex`, `reason_for_stop`, `type`, `county_name`) via `StringIndexer + OneHotEncoder`.
   * Encode `department_name` using the numeric `dept_freq` feature (count of stops per department).
   * Assemble numeric + encoded features → `features_raw`.
   * Standardize into `features` using `StandardScaler`.

5. Train models:

   * Logistic Regression (`LR_T0`, `LR_T1`) — always.
   * Gradient-Boosted Trees (`GBT_T0`, `GBT_T1`) — if `RUN_GBT = True`.

6. Evaluate on test set:

   * AUROC (`areaUnderROC`) and AUPRC (`areaUnderPR`).
   * Fairness-like summaries by `subject_race` and `subject_sex`:

     * TPR, FPR, selection rate.

7. Write models and metrics to GCS:

   For each label (e.g. `citation_issued`) and run timestamp:

   ```text
   gs://main-bucket-67/outputs/citation_issued/<timestamp>/
     models/
       LR_T0/
       LR_T1/
       GBT_T0/        # if RUN_GBT = True
       GBT_T1/        # if RUN_GBT = True
     metrics/json/
       part-*.txt     # single JSON string with all metrics + fairness info
   ```

The enriched data lives at:

```text
gs://main-bucket-67/outputs/enriched/nc/
```

---

## 4. Dataproc Cluster Configurations

You’ll run the same script on three cluster shapes:

1. **Single Node**

   * 1 node
   * 4 vCPUs, 16 GB RAM

2. **1 Master + 2 Workers**

   * Primary: 4 vCPUs, 16 GB
   * 2 workers: each 4 vCPUs, 16 GB

3. **1 Master + 4 Workers**

   * Primary: 4 vCPUs, 16 GB
   * 4 workers: each 4 vCPUs, 16 GB

### 4.1. Create cluster – general steps

For each configuration:

1. GCP Console → **Dataproc → Clusters → Create cluster**.
2. Choose region (e.g., `us-central1`) matching your bucket.
3. Choose **Cluster type**:

   * **Single node** OR **Standard** (for multi-node).
4. Customize machine types:

   * Click **Customize** for primary/worker:

     * vCPUs: **4**
     * Memory: **16 GB**
5. (Optional) Set Spark shuffle partitions in “Properties”:

   * `spark:spark.sql.shuffle.partitions=400` (or more for larger clusters).
6. Click **Create** and wait for the cluster to be ready.

Repeat to build:

* Cluster A: Single node.
* Cluster B: 1M + 2W.
* Cluster C: 1M + 4W.

---

## 5. Running `main.py` on a Cluster

For **each cluster configuration**, follow the same workflow.

### 5.1. SSH into the master

1. GCP Console → **Dataproc → Clusters** → click your cluster.
2. Go to the **VM instances** tab.
3. Click **SSH** for the master instance (e.g. `cluster-single-node-m`).

### 5.2. Sanity check PySpark

In the SSH terminal:

```bash
pyspark
```

You should see a Spark shell and a `spark` session. Then exit:

```python
exit()
```

### 5.3. Upload `main.py` to the master

Option A: from GCS (if you’ve uploaded it there):

```bash
gsutil cp gs://main-bucket-67/code/main.py .
```

Option B: via SSH web UI:

* In the SSH window, click the **gear icon → Upload file**.
* Select your local `main.py`.
* Confirm it’s there:

  ```bash
  ls
  head -n 30 main.py
  ```

Make sure `BUCKET` and `RAW_URI` at the top of `main.py` match your bucket and path.

### 5.4. Run the job

From the master SSH shell:

```bash
spark-submit --master yarn --deploy-mode cluster main.py
```

On the **first successful run** on any cluster:

* The script will see that `ENRICHED_PATH` does not exist.
* It will **build enrichment from raw CSV**, including ACS lookup.
* It will write enriched Parquet to `gs://main-bucket-67/outputs/enriched/nc`.

On subsequent runs:

* It will attempt to read from `ENRICHED_PATH` and **skip enrichment**.
* If the read fails (e.g., corrupted data), it will log a warning and rebuild from raw CSV automatically.

---

## 6. Monitoring Progress

While the job is running:

### 6.1. CLI via YARN

On the master SSH shell:

```bash
# See running applications
yarn application -list

# Get status of a specific application
yarn application -status <application_id>
```

This gives you:

* State (`RUNNING`, `FINISHED`, etc.).
* Duration.
* Tracking URL for the Spark UI.

To view logs so far:

```bash
yarn logs -applicationId <application_id> | less
```

You’ll see your `[INFO]` prints plus Spark logs.

### 6.2. Spark UI

1. In the GCP Console → **Dataproc → Clusters → your cluster**.
2. Click **Web interfaces**.
3. Open **YARN ResourceManager**.
4. Find your running application and click the **tracking URL / Spark UI**.

There you can see:

* Jobs and stages
* Task completion count
* Shuffle read/write
* Executors (CPU/memory usage)

---

## 7. Collecting Results and Comparing Clusters

For each cluster configuration:

1. Run `main.py` as shown.

2. Record **runtime**:

   * From `yarn application -status` or Dataproc Jobs UI.

3. Collect **model metrics** from GCS:

   ```text
   gs://main-bucket-67/outputs/citation_issued/<timestamp>/metrics/json/part-*.txt
   gs://main-bucket-67/outputs/arrest_made/<timestamp>/metrics/json/part-*.txt
   ```

   Each file holds a single JSON string with:

   * Model list:

     * `LR_T0_no_time_*`, `LR_T1_with_time_*`
     * `GBT_T0_no_time_*`, `GBT_T1_with_time_*` (if `RUN_GBT = True`)
   * AUROC & AUPRC
   * Fairness tables by race and sex
   * Class counts and weights.

4. Create a comparison table for your report, e.g.:

| Cluster Mode           | Label           | Model            | AUROC | AUPRC | Runtime (min) |
| ---------------------- | --------------- | ---------------- | ----- | ----- | ------------- |
| Single Node (4c/16GB)  | citation_issued | LR_T0_no_time    |       |       |               |
|                        |                 | LR_T1_with_time  |       |       |               |
|                        |                 | GBT_T0_no_time   |       |       |               |
|                        |                 | GBT_T1_with_time |       |       |               |
| 1M + 2W (4c/16GB each) | ...             | ...              |       |       |               |
| 1M + 4W (4c/16GB each) | ...             | ...              |       |       |               |

This lets you talk about:

* **Scaling out** (more nodes, same per-node specs) vs runtime and shuffle.
* **Scaling up sample size** (adjust `SAMPLE_FRAC` toward 1.0).
* **Effect of time features** (T0 vs T1) on predictive performance.

---

## 8. Resetting / Rerunning

* To force a **clean re-enrichment**:

  ```bash
  gsutil -m rm -r gs://main-bucket-67/outputs/enriched/nc
  ```

  Then rerun `spark-submit`.

* To run on a **smaller sample** for debugging:

  * Edit `SAMPLE_FRAC` at the top of `main.py`, e.g. `0.01`.
  * Re-upload `main.py` and re-run `spark-submit`.

* To test different **cluster shapes**:

  * Recreate the cluster with the desired nodes/cores/memory.
  * Repeat the SSH + `spark-submit` workflow.
