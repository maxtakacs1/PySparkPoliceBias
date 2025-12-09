# NC Open Policing – Spark ML Project on GCP Dataproc

This project uses **PySpark on Google Cloud Dataproc** to analyze the **North Carolina State Patrol** data from the Stanford **Open Policing Project** and:

* Enriches the data with **time features** and **county-level ACS demographics**.
* Trains two sets of models for two labels:

  * `citation_issued`
  * `arrest_made`
* For each label, compares:

  * **T0** – model **without time** features
  * **T1** – model **with time** features
* Runs and compares performance on three **cluster configurations**:

  1. **Single Node**: 4 cores, 24 GB RAM
  2. **1 Master, 2 Workers**: 4 cores, 24 GB RAM per node
  3. **1 Master, 4 Workers**: 4 cores, 24 GB RAM per node

Run mode is exactly as requested:

> Create Dataproc cluster → SSH into master VM → run `pyspark` (sanity check) → upload `main.py` → run
> `spark-submit --master yarn --deploy-mode cluster main.py`.

---

## 1. Prerequisites

* A **GCP project** with billing enabled.
* Permissions to:

  * Create **Cloud Storage** buckets.
  * Create **Dataproc clusters**.
  * SSH into Dataproc **VM instances**.
* `yg821jf8611_nc_statewide_2020_04_01.csv.zip` (NC statewide Open Policing dataset).
* (Optional but recommended) a **Census API key** for ACS:

  * Can be requested from the US Census:
    [https://www.census.gov/data/developers/guidance/api-user-guide.html](https://www.census.gov/data/developers/guidance/api-user-guide.html)

Dataproc images already include:

* `pyspark`
* `spark-submit`
* `gsutil`

No extra installation is required on the cluster.

---

## 2. Cloud Storage layout

1. In the GCP Console, go to:
   **Navigation ☰ → Cloud Storage → Buckets → Create**
2. Create a bucket (example):
   `your-nc-opp-bucket`

   * Region: choose one (e.g., `us-central1`). Remember this for Dataproc.
3. Inside the bucket, create folders:

```text
gs://your-nc-opp-bucket/
  raw/
  code/      # optional – to store main.py
  outputs/   # will be created/used by the job
```

4. Upload the raw ZIP:

```text
raw/yg821jf8611_nc_statewide_2020_04_01.csv.zip
```

---

## 3. `main.py` – Configuration and Overview

The main driver script is `main.py`. It is intended to be run via:

```bash
spark-submit --master yarn --deploy-mode cluster main.py
```

### 3.1. Configuration constants

At the top of `main.py`, edit only the constants in the **CONFIG** section:

```python
# GCS bucket (must include gs://)
BUCKET = "gs://your-nc-opp-bucket"  # <-- CHANGE THIS

# RAW ZIP from Open Policing Project (already uploaded to your bucket)
RAW_URI = f"{BUCKET}/raw/yg821jf8611_nc_statewide_2020_04_01.csv.zip"

# Enriched Parquet location (directory)
ENRICHED_PATH = f"{BUCKET}/outputs/enriched/nc"

# Outputs base (models + metrics)
OUTPUT_BASE = f"{BUCKET}/outputs"

# Optional Census API key; None = anonymous (OK for small pulls)
CENSUS_API_KEY = None  # e.g., "YOUR_CENSUS_KEY"

# Training knobs (no CLI args)
SEED          = 42
SAMPLE_FRAC   = 1.0          # set <1.0 for quick subsample runs, e.g. 0.05
RUN_GBT       = True         # also train GBT models
HASH_DEPT     = True         # hash department_name
NUM_HASH_FEAT = 1 << 18      # 262,144 hashed dims
```

Everything else in `main.py` is ready to go.

### 3.2. What the script does

On **first run**:

1. **Enrichment step**

   * Reads `RAW_URI` (ZIP) from GCS.
   * On the Dataproc master:

     * Copies ZIP to `/tmp`
     * Extracts the first CSV
     * Uploads the CSV back to GCS (next to the ZIP).
   * Loads CSV with Spark and:

     * Cleans columns and handles `"NaN"` / empty strings.
     * Normalizes boolean-like fields (`true/false`, `yes/no`, etc.) into `0/1`.
     * Clips `subject_age` to [16, 100], nulls anything outside.
     * Builds **time features**:

       * `time_available`, `hour_of_day`, `day_of_week`, `is_weekend`
       * `hour_sin`, `hour_cos`
     * Creates labels and flags:

       * `citation_yn`, `arrest_yn`
       * `outcome_multiclass` (arrest/citation/warning/other)
       * `location_known`, `county_known`
       * Normalized `county_name_norm` for ACS joining.
     * Computes initial global `dept_freq` from `department_name`.
     * Calls the **Census ACS API** for NC counties and joins:

       * `median_income`
       * `poverty_rate`
       * `unemployment_rate`
       * `edu_bachelor_pct`
   * Writes partitioned **enriched Parquet** to `ENRICHED_PATH`.

On **subsequent runs**:

* If `ENRICHED_PATH` exists in GCS, the script **skips enrichment** and directly loads it.

2. **Modeling step**

For each label in:

* `citation_issued`
* `arrest_made`

it:

* Filters to rows where `time_available == 1` (to compare with vs without time on same cohort).
* Performs an **80/20 train/test random split** (fixed seed).
* Computes **class weights** (`weightCol = class_w`) for class imbalance.
* Builds and trains:

  * **T0** (no time features):

    * Demographics, stop context, county & ACS, department info.
  * **T1** (with time features):

    * All T0 features + `time_available`, `hour_of_day`, `hour_sin`, `hour_cos`, `is_weekend`.
  * Algorithms:

    * Logistic Regression (always)
    * Gradient-Boosted Trees (if `RUN_GBT = True`)
* Evaluates:

  * AUROC, AUPRC for each model.
  * Fairness snapshots (TPR, FPR, selection rate) by `subject_race` and `subject_sex`.

3. **Outputs**

For each label, writing under:

```text
gs://your-nc-opp-bucket/outputs/<label>/<timestamp>/
  models/
    LR_T0/
    LR_T1/
    GBT_T0/ (if RUN_GBT)
    GBT_T1/ (if RUN_GBT)
  metrics/json/
    part-*.txt   # contains a single JSON string with metrics + fairness tables
```

The enriched dataset lives at:

```text
gs://your-nc-opp-bucket/outputs/enriched/nc/
```

---

## 4. Creating and running on each cluster configuration

You will **repeat this section three times**, once per cluster configuration:

1. Single Node (4 cores, 24 GB)
2. 1 Master + 2 Workers (each 4 cores, 24 GB)
3. 1 Master + 4 Workers (each 4 cores, 24 GB)

### 4.1. Create cluster – Single Node (4 cores, 24 GB)

1. Console → **Dataproc → Clusters → Create cluster**.
2. Region: same as your bucket (e.g., `us-central1`).
3. **Cluster type**: **Single node**.
4. Under **Configure nodes → Primary node → Machine type → Customize**:

   * Set **vCPUs = 4**
   * Set **Memory = 24 GB**
5. Under **Components**, enable **Component Gateway** (optional).
6. Under **Properties** (expand “Customize cluster” → “Properties”), you can add:

   * `spark:spark.sql.shuffle.partitions=200`
7. Click **Create** and wait for cluster to be ready.

### 4.2. Create cluster – 1 Master + 2 Workers (4 cores, 24 GB each)

1. Console → **Dataproc → Clusters → Create cluster**.
2. Region: same as bucket.
3. **Cluster type**: **Standard**.
4. **Primary node**:

   * Machine type → **Customize** → 4 vCPUs, 24 GB.
5. **Worker nodes**:

   * Number of workers: **2**
   * Machine type → **Customize** → 4 vCPUs, 24 GB.
6. Properties:

   * `spark:spark.sql.shuffle.partitions=400`
7. Click **Create**.

### 4.3. Create cluster – 1 Master + 4 Workers (4 cores, 24 GB each)

As above, but:

* Number of workers: **4**
* Suggested property:

  * `spark:spark.sql.shuffle.partitions=800`

---

## 5. SSH Workflow and Running the Job

Repeat this for each cluster configuration.

### 5.1. SSH into master VM

1. Console → **Dataproc → Clusters** → click your cluster.
2. Click the **VM instances** tab.
3. For the master instance (e.g., `cluster-name-m`), click **SSH**.

### 5.2. Run `pyspark` (sanity check)

In the SSH terminal:

```bash
pyspark
```

You should see a PySpark shell start and a `SparkSession` named `spark`.

Type `spark` and confirm it prints a SparkSession object. Then exit:

```python
exit()
```

### 5.3. Copy `main.py` onto the master

Option A – If you uploaded `main.py` to GCS:

```bash
gsutil cp gs://your-nc-opp-bucket/code/main.py .
```

Option B – Paste manually on the VM:

```bash
nano main.py
# paste the entire script from your editor
# Ctrl+O (write out), Enter, Ctrl+X (exit)
```

Make sure the `BUCKET` constant at the top of `main.py` matches your bucket.

### 5.4. Run the job via `spark-submit`

From the same SSH shell:

```bash
spark-submit --master yarn --deploy-mode cluster main.py
```

What this does:

* Submits a Spark application to YARN.
* Driver runs in a YARN container; your SSH session just monitors logs.
* On the **first** cluster you run it on (e.g., Single Node), it will:

  * Extract & enrich raw data (including ACS calls).
  * Write enriched Parquet to `ENRICHED_PATH`.
* On subsequent clusters, the enrichment step is **skipped** and the script directly loads the enriched dataset.

You’ll see a YARN application ID printed (e.g., `application_...`).

You can check status from the SSH shell:

```bash
yarn application -list
yarn application -status <application_id>
```

Or via the Web UI (Component Gateway → YARN ResourceManager).

---

## 6. Collecting Results and Performance Metrics

For your report, you want to compare:

* Runtime and resource usage across cluster configurations.
* Model performance and fairness.

### 6.1. Runtime and resource usage

For each cluster configuration:

1. Note the **start** and **finish** times from:

   * `yarn application -status <app_id>` (fields `Start-Time` and `Finish-Time`), or
   * Dataproc → **Jobs** → Spark job → `Duration`.
2. In the **YARN ResourceManager UI**:

   * Look at the application’s **attempt**.
   * Record:

     * Number of **executors** used.
     * **Memory per executor**.
     * **Cores per executor**.
     * Shuffle read/write (MB/GB).

You can then compute a simple **cost proxy**:
`cluster size (#nodes, cores) × runtime (hours)`.

### 6.2. Model performance and fairness

After each run, `main.py` writes:

```text
gs://your-nc-opp-bucket/outputs/citation_issued/<timestamp>/
gs://your-nc-opp-bucket/outputs/arrest_made/<timestamp>/
  models/
  metrics/json/
```

1. Go to **Cloud Storage → your bucket → outputs/**.
2. For each label (`citation_issued`, `arrest_made`) and timestamp:

   * Navigate to `metrics/json`.
   * Download the `part-*.txt` file.
   * It contains a single JSON string with:

     * `results` (a list of models and their metrics: AUROC, AUPRC, fairness by race/sex).
     * `class_counts_train` (pos/neg totals).
     * `weights` (class weights used).
3. Use a local notebook or Python script to parse and compare:

   * **T0 vs T1** for each algorithm (LR, GBT).
   * Across cluster shapes, metrics should be very similar; the main difference is runtime and resource usage.

You can build tables like:

| Cluster Mode          | Label           | Model            | AUROC | AUPRC | Runtime (min) |
| --------------------- | --------------- | ---------------- | ----- | ----- | ------------- |
| Single Node (4c/24GB) | citation_issued | LR_T0_no_time    |       |       |               |
|                       |                 | LR_T1_with_time  |       |       |               |
|                       |                 | GBT_T0_no_time   |       |       |               |
|                       |                 | GBT_T1_with_time |       |       |               |
| 1M + 2W (4c/24GB)     | ...             | ...              |       |       |               |
| 1M + 4W (4c/24GB)     | ...             | ...              |       |       |               |

---

## 7. Resetting or Rerunning

* If you want to force a fresh **enrichment**, delete the folder:

```bash
gsutil -m rm -r gs://your-nc-opp-bucket/outputs/enriched/nc
```

Then re-run `spark-submit`.

* If you only want to rerun modeling with different sampling (`SAMPLE_FRAC`) or other training knobs, just edit the constants at the top of `main.py` and re-run (the enrichment will be reused).

---

## 8. Summary

To run everything:

1. **Prepare**:

   * Upload the ZIP to `gs://your-nc-opp-bucket/raw/`.
   * Edit and upload/copy `main.py` with correct `BUCKET`.

2. **Create cluster** in one of the three configurations.

3. **SSH → `pyspark`** (sanity check), exit.

4. **Upload `main.py`** to master.

5. Run:

   ```bash
   spark-submit --master yarn --deploy-mode cluster main.py
   ```

6. **Repeat** steps 2–5 for each cluster configuration (Single Node, 1M+2W, 1M+4W).

7. Collect **runtime**, **resource usage**, and **metrics JSON** from GCS for your analysis and presentation.