# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install dependencies
# MAGIC %pip install /Volumes/dev_catalog/shared/libs/openpyxl-3.1.5-py2.py3-none-any.whl /Volumes/dev_catalog/shared/libs/et_xmlfile-2.0.0-py3-none-any.whl -q

# COMMAND ----------

# DBTITLE 1,Import sql_pyspark and schema_loader
# MAGIC %run ../common/sql_pyspark
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import silver_incremental
# MAGIC %run ../common/silver_incremental

# COMMAND ----------

# MAGIC %run ../common/schema_loader

# COMMAND ----------

# DBTITLE 1,Import modules
# MAGIC %run ../common/qa

# COMMAND ----------

# DBTITLE 1,Import source_code_validation
# MAGIC %run ../common/source_code_validation

# COMMAND ----------

# DBTITLE 1,Import parser
# MAGIC %run ../common/parser

# COMMAND ----------

# DBTITLE 1,Import utilities
# MAGIC %run ../common/utilities

# COMMAND ----------

# DBTITLE 1,Import clients_loader
# MAGIC %run ../common/clients_loader

# COMMAND ----------

# DBTITLE 0,Import base_table_clients
# MAGIC %run ../common/base_table_clients

# COMMAND ----------

# DBTITLE 1,PySpark Imports
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, coalesce, upper, lower, trim, concat, substring,
    to_timestamp, to_date, year, month, dayofmonth, datediff, date_format,
    regexp_replace, split, monotonically_increasing_id, row_number, dense_rank,
    first, last, max as spark_max, min as spark_min, sum as spark_sum, 
    count as spark_count, countDistinct, collect_list, array, struct, expr,
    round as spark_round, abs as spark_abs, length, instr, locate,
    date_trunc, add_months, months_between, date_add, date_sub,
    array_contains, explode, posexplode, size, concat_ws,
    create_map, map_keys, map_values, element_at
)
from pyspark.sql.window import Window
from pyspark.sql.types import *
import json
import gc
import logging
import time

# COMMAND ----------

# DBTITLE 1,Import utilities and helpers
import traceback
from datetime import datetime, date
import ast

# Wall-clock between major steps (notebook output + log file). Enable via PHASE_TIMING=True or env BASE_TABLE_PHASE_TIMING=1
PHASE_TIMING = False
_PHASE_CLOCK = {'t0': None, 'prev': None}


def _phase_timing_on():
    import os
    if PHASE_TIMING:
        return True
    return os.environ.get('BASE_TABLE_PHASE_TIMING', '').strip().lower() in ('1', 'true', 'yes', 'on')


def _phase_reset():
    _PHASE_CLOCK['t0'] = None
    _PHASE_CLOCK['prev'] = None


def _log_phase(label, client=None):
    if not _phase_timing_on():
        return
    now = time.perf_counter()
    cl = client if client is not None else ''
    if _PHASE_CLOCK['t0'] is None:
        _PHASE_CLOCK['t0'] = now
        _PHASE_CLOCK['prev'] = now
        dt, tot = 0.0, 0.0
    else:
        dt = now - _PHASE_CLOCK['prev']
        tot = now - _PHASE_CLOCK['t0']
        _PHASE_CLOCK['prev'] = now
    msg = f'[BASE_TABLE_TIMING] {label} +{dt:.2f}s (total {tot:.2f}s) client={cl}'
    print(msg)
    logging.getLogger().warning(msg)


# Track temp Delta tables for cleanup
_TEMP_TABLES = []
_TEMP_TABLE_PREFIX = '_btg_tmp_'


def _safe_cache(df, label='cache'):
    """Materialize DataFrame to a temp Delta table, breaking lineage completely.
    Works on all compute environments (classic cluster, serverless, shared).

    Why temp Delta instead of .cache():
    - .cache() can be evicted on classic clusters — if the source data was
      mutated (e.g. rows deleted), re-execution returns wrong results.
    - .cache() is unreliable on serverless — executors can be replaced,
      losing cached data.
    - Temp Delta writes to durable cloud storage (ADLS/S3), survives
      executor replacement and memory pressure.
    - spark.table() returns a new DataFrame with no lineage to the
      original query plan.

    Temp tables are cleaned up at pipeline end by _cleanup_temp_tables().
    Falls back to .cache() only if Delta write fails.
    """
    try:
        spark = SparkSession.getActiveSession()
        catalog = spark.catalog.currentCatalog()
        schema = spark.catalog.currentDatabase()
        tmp_name = f"{catalog}.{schema}.{_TEMP_TABLE_PREFIX}{label}_{int(time.time())}"
        df.write.mode("overwrite").format("delta").saveAsTable(tmp_name)
        _TEMP_TABLES.append(tmp_name)
        logger.info(f"_safe_cache: materialized to {tmp_name}")
        return spark.table(tmp_name)
    except Exception as e:
        logger.warning(f"_safe_cache: temp Delta failed ({e}), falling back to cache()")
        try:
            return df.cache()
        except Exception:
            return df


def _safe_unpersist(df):
    """Unpersist DataFrame. Temp Delta tables are cleaned up by _cleanup_temp_tables."""
    if df is None:
        return
    # Temp Delta tables are cleaned up at end of pipeline via _cleanup_temp_tables.
    # No per-call cleanup needed.
    return


def _cleanup_temp_tables():
    """Drop all temp Delta tables created during the pipeline."""
    spark = SparkSession.getActiveSession()
    for t in _TEMP_TABLES:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {t}")
            logger.info(f"Dropped temp table {t}")
        except Exception:
            pass
    _TEMP_TABLES.clear()


def _cleanup_stale_temp_tables():
    """Drop orphaned _btg_tmp_* tables from previous failed/interrupted runs.
    Only targets tables with our unique prefix — won't touch other temp tables.
    """
    try:
        spark = SparkSession.getActiveSession()
        catalog = spark.catalog.currentCatalog()
        schema = spark.catalog.currentDatabase()
        tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema} LIKE '{_TEMP_TABLE_PREFIX}*'").collect()
        if not tables:
            return
        for row in tables:
            tbl_name = row['tableName']
            if tbl_name.startswith(_TEMP_TABLE_PREFIX):
                full_name = f"{catalog}.{schema}.{tbl_name}"
                spark.sql(f"DROP TABLE IF EXISTS {full_name}")
                print(f"[STALE CLEANUP] Dropped orphaned temp table: {full_name}")
    except Exception as e:
        print(f"[STALE CLEANUP] Warning: could not clean stale temp tables: {e}")

# COMMAND ----------

# DBTITLE 1,Helper: create_schema_from_dict
# Module-level type map: JSON type string → PySpark type
# Used by create_schema_from_dict and process_trx_prep_data_pyspark
SPARK_TYPE_MAP = {
    'StringType': StringType(),
    'LongType': LongType(),
    'IntegerType': IntegerType(),
    'DoubleType': DoubleType(),
    'FloatType': FloatType(),
    'TimestampType': TimestampType(),
    'TIMESTAMP_LTZ': TimestampType(),  # Timestamp with Local TimeZone
    'TimestampNTZType': TimestampNTZType(),  # Timestamp without TimeZone
    'DateType': DateType(),
    'BooleanType': BooleanType(),
    'DecimalType': DecimalType(18, 2),
}


def create_schema_from_dict(columns, type_dict):
    """Create PySpark StructType schema from column list and type dictionary."""
    fields = []
    for col_name in columns:
        type_string = type_dict.get(col_name)
        if type_string is None:
            print(f"Warning: Column '{col_name}' has no type definition, defaulting to StringType")
            spark_type = StringType()
        elif type_string not in SPARK_TYPE_MAP:
            print(f"Warning: Unknown type '{type_string}' for column '{col_name}', defaulting to StringType")
            spark_type = StringType()
        else:
            spark_type = SPARK_TYPE_MAP[type_string]
        fields.append(StructField(col_name, spark_type, nullable=True))
    
    return StructType(fields)

# COMMAND ----------

# DBTITLE 1,Helper: database_headers_pyspark
def database_headers_pyspark(df):
    """
    PySpark equivalent of database_headers (from sql.py:292)
    Convert CamelCase to snake_case and clean column names.
    Handles duplicate normalized names by coalescing columns.
    """
    import re
    
    original_cols = list(df.columns)
    new_columns = []
    for col_name in original_cols:
        # Underscore camel case conversion
        n = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col_name)
        n = re.sub('([a-z0-9])([A-Z])', r'\1_\2', n)
        n = n.lower()
        
        # Remove special characters
        symbols = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', 
                   '*', '|', ',', '&', '<', '`', '}', '.', '=', ']', '!', '>', ';', 
                   '?', '$', ')', '/', ' ', '\u00b7']
        for ch in symbols:
            n = n.replace(ch, '_')
        n = n.replace('#', 'nmb')
        n = n.replace('__', '_')
        n = n.rstrip('_')
        
        new_columns.append(n)
    
    # Detect duplicate normalized names and coalesce before renaming
    from collections import Counter
    name_counts = Counter(new_columns)
    duplicates = {n for n, cnt in name_counts.items() if cnt > 1}
    
    drop_originals = set()  # original column names to drop
    if duplicates:
        for dup_name in sorted(duplicates):
            # Find all original columns that map to this duplicate name
            indices = [i for i, n in enumerate(new_columns) if n == dup_name]
            orig_names = [original_cols[i] for i in indices]
            logger.info(f'[DB_HEADERS] Duplicate normalized name "{dup_name}" from columns: {orig_names} \u2014 coalescing')
            # Coalesce: prefer the first column's value, fallback to subsequent
            coalesce_cols = [col(f'`{original_cols[i]}`') for i in indices]
            keep_orig = original_cols[indices[0]]
            df = df.withColumn(keep_orig, coalesce(*coalesce_cols))
            # Mark duplicates (all except first) for dropping
            for i in indices[1:]:
                drop_originals.add(original_cols[i])
    
    # Drop duplicate source columns
    if drop_originals:
        for dc in drop_originals:
            df = df.drop(df[dc])
    
    # Rebuild column lists after drops
    remaining_originals = [c for c in original_cols if c not in drop_originals]
    remaining_new = [new_columns[i] for i, c in enumerate(original_cols) if c not in drop_originals]
    
    # Rename all columns
    for old_col, new_col in zip(remaining_originals, remaining_new):
        if old_col != new_col:
            df = df.withColumnRenamed(old_col, new_col)
    
    return df

# COMMAND ----------

# DBTITLE 1,Helper: fiscal_from_column_pyspark
def fiscal_from_column_pyspark(df, date_col, first_month, output_col=None):
    """
    PySpark equivalent of fiscal_from_column (utilities.py:602)
    Calculate fiscal year based on date column and fiscal year start month.

    Logic (matches Pandas original):
      - If first_month == 1: FY = year
      - If first_month != 1:
          - month >= first_month → year + 1
          - month <  first_month → year

    Args:
        df: Spark DataFrame
        date_col: Name of date column
        first_month: Starting month of fiscal year (1-12)
        output_col: Name for output column (optional, returns expression if None)

    Returns:
        DataFrame with new column if output_col specified, else expression
    """
    if first_month != 1:
        fy_expr = when(month(col(date_col)) >= first_month,
                       year(col(date_col)) + 1).otherwise(year(col(date_col)))
    else:
        fy_expr = year(col(date_col))

    if output_col:
        return df.withColumn(output_col, fy_expr)
    else:
        return fy_expr

# COMMAND ----------

# DBTITLE 1,Helper: remove_client_prefix_pyspark
def remove_client_prefix_pyspark(df, client):
    """
    Remove client prefix from column names.
    If removing the prefix would collide with an existing column,
    drop the existing column first (the prefixed FH filter version
    is the canonical cleaned/computed column).
    """
    prefix = client.upper() + '_'
    prefix_lower = client.lower() + '_'
    
    for col_name in df.columns:
        if col_name.startswith(prefix) or col_name.startswith(prefix_lower):
            new_name = col_name[len(prefix):]
            if new_name in df.columns:
                df = df.drop(new_name)
            df = df.withColumnRenamed(col_name, new_name)
    
    return df

# COMMAND ----------

# DBTITLE 1,def clean_bronze_column_names_pyspark
def clean_bronze_column_names_pyspark(df, bronze_renames):
    """
    Clean up bronze column names for silver layer.
    Removes trailing underscores and applies standardization.
    
    Args:
        df: PySpark DataFrame from bronze
        bronze_renames: Dictionary of bronze_col → silver_col mappings
    
    Returns:
        DataFrame with cleaned column names
    """
    for bronze_col, silver_col in bronze_renames.items():
        if bronze_col in df.columns:
            df = df.withColumnRenamed(bronze_col, silver_col)
            print(f"[RENAME] Renamed bronze column: {bronze_col} -> {silver_col}")
    
    return df

# COMMAND ----------

# DBTITLE 1,Helper: _clean_timestamp_columns
def _clean_timestamp_columns(df):
    """
    Clean STRING columns that represent dates: convert empty strings to NULL and cast to TimestampNTZType
    Note: Columns already in TimestampType don't need cleaning (no empty strings, only NULL or valid timestamps)
    
    Args:
        df: PySpark DataFrame from bronze
    
    Returns:
        DataFrame with cleaned timestamp columns (as TimestampNTZ)
    """
    for field in df.schema.fields:
        col_name = field.name
        # Only process STRING columns that likely contain date data
        if str(field.dataType) == 'StringType' and 'date' in col_name.lower():
            # Convert empty strings to NULL, then cast to timestamp (coerce failures to NULL)
            df = df.withColumn(col_name,
                coalesce(
                    to_timestamp(
                        when((trim(col(col_name)) == '') | (col(col_name).isNull()),
                             lit(None))
                        .otherwise(col(col_name))
                    ),
                    lit(None).cast(TimestampNTZType())
                )
            )
    return df

# COMMAND ----------

# DBTITLE 1,Helper: load_sc_pyspark
def load_sc_pyspark(client):
    """
    Load source code from bronze Delta table - returns Spark DataFrame
    Pandas version: load_sc() in utilities.py:1933
    """
    df = _read_from_bronze_pyspark(client, 'sourcecode')
    if df is not None:
        df = _clean_timestamp_columns(df)
        df = df.withColumn('Client', lit(client))
        return df
    print(f"load_sc_pyspark({client}): bronze sourcecode table not available")
    return None

# COMMAND ----------

# DBTITLE 1,Helper: load_parquet_pyspark
def load_parquet_pyspark(client, last_watermark=None):
    """
    Load raw gift data from bronze Delta table - returns Spark DataFrame
    Pandas version: load_parquet() in utilities.py:1873

    Args:
        client: Client identifier
        last_watermark: Optional watermark for incremental read (filters _update_ts > watermark)
    """
    df = _read_from_bronze_pyspark(client, 'gift', last_watermark=last_watermark)
    if df is not None:
        df = _clean_timestamp_columns(df)
        
        # Handle both GiftDate and gift_date column names (create GiftDate alias if needed)
        if 'gift_date' in df.columns and 'GiftDate' not in df.columns:
            df = df.withColumn('GiftDate', col('gift_date'))
        
        df = df.withColumn('Client', lit(client))
        return df
    print(f"load_parquet_pyspark({client}): bronze table not available")
    return None

# COMMAND ----------

# DBTITLE 1,Helper: load_budget_pyspark
def load_budget_pyspark(client):
    """
    Load budget from bronze Delta table - returns Spark DataFrame
    Pandas version: load_budget() in utilities.py:1943
    """
    df = _read_from_bronze_pyspark(client, 'budget')
    if df is not None:
        df = _clean_timestamp_columns(df)
        return df
    print(f"load_budget_pyspark({client}): bronze budget table not available")
    return None

# COMMAND ----------

# DBTITLE 1,Helper: _read_from_bronze_pyspark
def _read_from_bronze_pyspark(client, table_type, last_watermark=None):
    """
    Read from bronze Delta table - returns Spark DataFrame
    Pandas version: _read_from_bronze() in utilities.py:1857

    Args:
        client: Client identifier
        table_type: Type of table ('gift', 'sourcecode', 'budget')
        last_watermark: Optional watermark timestamp for incremental read.
                        When provided, filters on _update_ts > last_watermark.
    """
    try:
        cfg = _load_etl_config()
        catalog = cfg.get('bronze_catalog', cfg.get('metadata_catalog', 'dev_catalog'))
        suffix_map = cfg.get('table_suffix_map', {})
        suffix = suffix_map.get(table_type, table_type)
        client_schema = client.lower()
        # New naming convention: no client prefix in table name
        # e.g., dev_catalog.care.gift_bronze (not dev_catalog.care.care_gift_bronze)
        table_name = f"{catalog}.{client_schema}.{suffix}"
        
        spark = _get_spark()
        df = spark.table(table_name)
        
        # Incremental filter: only read rows newer than last watermark
        if last_watermark is not None and '_update_ts' in df.columns:
            df = df.filter(
                (col('_update_ts') > lit(last_watermark)) | col('_update_ts').isNull()
            )
            _row_count = df.count()
            logger.info(f'[BRONZE READ] {table_name}: INCREMENTAL (_update_ts > {last_watermark}) \u2192 {_row_count:,} rows')
        elif last_watermark is not None:
            logger.warning(f'[BRONZE READ] {table_name}: FULL (no _update_ts column, watermark ignored)')
        else:
            logger.info(f'[BRONZE READ] {table_name}: FULL')
        
        return df
    except Exception as e:
        logger.warning(f"[BRONZE READ] {client}.{table_type}: table not found or read error \u2014 {e}")
        return None

# COMMAND ----------

# DBTITLE 1,notification_exception_raised
def notification_exception_raised(client, code_location, error_message):
    """Log errors to ETL2 status table and logger.
    Email notifications removed — using Databricks workflow alerts instead.
    """
    etl2_status_entry(client, f'[RAISED ERROR] {code_location}: {error_message}')
    logger.error(f'{code_location}: {error_message}')

# COMMAND ----------

# DBTITLE 1,def sc_load_and_process_csv_pyspark
def sc_load_and_process_csv_pyspark(client, max_gift, min_gift):
    """
    PySpark version - Original lines 153-373
    Load and process source code data
    """
    schema = get_schema(client)
    spark = _get_spark()
    
    ################################################
    # Read source code from bronze Delta table
    ################################################
    sc = load_sc_pyspark(client)
    if sc is None:
        raise Exception(f"Failed to load SourceCode for client {client}")
    
    # Clean bronze column names for silver layer
    sc = clean_bronze_column_names_pyspark(sc, bronze_to_silver_renames)
    
    sc = apply_client_source_code_transform(sc, client)
    
    ################################################
    # Format sourcecode
    ################################################
    
    ### WORKAROUNDS for inconsistencies
    # Pandas: sc.columns = sc.columns.str.replace(' ', '')
    for old_col in sc.columns:
        new_col = old_col.replace(' ', '')
        if old_col != new_col:
            sc = sc.withColumnRenamed(old_col, new_col)
    
    # Check if all values in 'MailDate' are null (limit(1) avoids full scan)
    # Pandas: if sc['MailDate'].isnull().all()
    if sc.filter(col('MailDate').isNotNull()).limit(1).count() == 0:
        sc = sc.withColumn('MailDate', lit(None).cast(TimestampNTZType()))
    
    # Formats
    # Pandas: sc.CampaignCode = sc.CampaignCode.astype(str)
    sc = sc.withColumn('CampaignCode', col('CampaignCode').cast(StringType()))
    
    # Combine campaign code columns if there are multiple
    if 'CampaignCode' in sc.columns and 'Campaign_Code' in sc.columns:
        # Pandas: sc['CampaignCode']=sc['CampaignCode'].fillna(sc['Campaign Code'])
        sc = sc.withColumn('CampaignCode', coalesce(col('CampaignCode'), col('Campaign_Code')))
        # Pandas: sc.drop('Campaign Code', axis=1, inplace=True)
        sc = sc.drop('Campaign_Code')
    
    if 'SourceCode' in sc.columns and 'Source_Code' in sc.columns:
        sc = sc.withColumn('SourceCode', coalesce(col('SourceCode'), col('Source_Code')))
        sc = sc.drop('Source_Code')
    
    # Add filters that apply only to the dimension (source code) table
    sc = add_dimension_filters(sc, client)
    
    # Convert MailDate to datetime
    try:
        # Attempt the first method
        # Pandas: sc['MailDate'] = pd.to_datetime(sc['MailDate'].str[:10])
        sc = sc.withColumn('MailDate', to_timestamp(substring(col('MailDate'), 1, 10)).cast(TimestampNTZType()))
    except Exception as e:
        print(f"Error encountered: {e}. Falling back to alternate method.")
        
        # Ensure MailDate is a string and strip spaces
        # Pandas: sc['MailDate'] = sc['MailDate'].astype(str).str.strip()
        sc = sc.withColumn('MailDate', trim(col('MailDate').cast(StringType())))
        
        # Remove time component if present (keeps only date part)
        # Pandas: sc['MailDate'] = sc['MailDate'].str.split(" ").str[0]
        sc = sc.withColumn('MailDate', split(col('MailDate'), ' ').getItem(0))
        
        # Try parsing MM/DD/YYYY first
        # Pandas: maildate_mdY = pd.to_datetime(sc['MailDate'], format="%m/%d/%Y", errors='coerce')
        sc = sc.withColumn('maildate_mdY', to_timestamp(col('MailDate'), 'MM/dd/yyyy').cast(TimestampNTZType()))
        
        # Try parsing YYYY-MM-DD next
        # Pandas: maildate_Ymd = pd.to_datetime(sc['MailDate'], format="%Y-%m-%d", errors='coerce')
        sc = sc.withColumn('maildate_Ymd', to_timestamp(col('MailDate'), 'yyyy-MM-dd').cast(TimestampNTZType()))
        
        # Merge results, prioritizing known formats
        # Pandas: sc['MailDate'] = maildate_mdY.fillna(maildate_Ymd)
        sc = sc.withColumn('MailDate', coalesce(col('maildate_mdY'), col('maildate_Ymd')))
        
        # Convert all valid dates to MM/DD/YYYY format
        # Pandas: sc.loc[sc['MailDate'].notna(), 'MailDate'] = sc['MailDate'].dt.strftime("%m/%d/%Y")
        sc = sc.withColumn('MailDate',
            when(col('MailDate').isNotNull(), date_format(col('MailDate'), 'MM/dd/yyyy'))
            .otherwise(col('MailDate')))
        
        # Drop temporary columns
        sc = sc.drop('maildate_mdY', 'maildate_Ymd')
    
    # Add mail_date_original column before missing are filled
    # Pandas: sc['mail_date_original'] = sc['MailDate']
    sc = sc.withColumn('mail_date_original', col('MailDate'))
    
    # Add fy_mail_date_original column
    # Pandas: sc['fy_mail_date_original'] = fiscal_from_column(sc, 'MailDate', schema['firstMonthFiscalYear'])
    sc = sc.withColumn('fy_mail_date_original',
        fiscal_from_column_pyspark(sc, 'MailDate', schema['firstMonthFiscalYear']))
    
    # New Cols
    # Pandas: sc['fy'] = fiscal_from_column(sc, 'MailDate', schema['firstMonthFiscalYear'])
    sc = sc.withColumn('fy',
        fiscal_from_column_pyspark(sc, 'MailDate', schema['firstMonthFiscalYear']))
    
    # Pandas: sc['DaysSinceMailDate'] = (max_gift - sc['MailDate']).dt.days
    sc = sc.withColumn('DaysSinceMailDate', datediff(lit(max_gift), col('MailDate')))
    
    # Keep only sourcecodes with mail date
    # Pandas: fy_valid_mask = sc['fy'] > 1900
    # Pandas: sc = sc.loc[fy_valid_mask]
    sc = sc.filter(col('fy') > 1900)
    
    # Backup Campaign Name
    # Pandas: sc['_CN'] = 'FY' + (sc['fy'].astype(str).replace('<NA>','XX').str[:4].str[-2:]).fillna('XX') + '_' + sc.MailDate.dt.month_name().fillna('') + '_' + sc.PackageName.fillna('')
    sc = sc.withColumn('_CN',
        concat(
            lit('FY'),
            substring(
                when(col('fy').isNotNull(),
                     regexp_replace(col('fy').cast(StringType()), '<NA>', 'XX'))
                .otherwise(lit('XX')),
                3, 2  # Get last 2 of first 4 characters
            ),
            lit('_'),
            coalesce(date_format(col('MailDate'), 'MMMM'), lit('')),
            lit('_'),
            coalesce(col('PackageName'), lit(''))
        ))
    
    # Pandas: sc.CampaignName = sc.CampaignName.fillna(sc._CN)
    sc = sc.withColumn('CampaignName', coalesce(col('CampaignName'), col('_CN')))
    
    # Pandas: sc = sc.drop('_CN', axis=1)
    sc = sc.drop('_CN')
    
    # --- Misc setup -------------------------------------------------------
    # Pandas: sc["Client"] = client
    sc = sc.withColumn('Client', lit(client))
    
    print("data_processed_at_og_sc started")
    
    # Ensure data_processed_at exists (already cleaned and cast to TimestampType in load_sc_pyspark)
    if 'data_processed_at' not in sc.columns:
        # Pandas: sc["data_processed_at"] = pd.NaT
        sc = sc.withColumn('data_processed_at', lit(None).cast(TimestampType()))
    
    # Fill nulls with the max non-null timestamp or current time
    # Pandas: if sc["data_processed_at"].notna().any()
    if sc.filter(col('data_processed_at').isNotNull()).limit(1).count() > 0:
        # Pandas: max_ts = sc["data_processed_at"].max()
        max_ts = sc.select(spark_max('data_processed_at').alias('_mts')).first()['_mts']
    else:
        # Pandas: max_ts = pd.to_datetime(ts_now(), errors="coerce")
        max_ts = ts_now()
    
    # Pandas: sc.loc[sc["data_processed_at"].isna(), "data_processed_at"] = max_ts
    sc = sc.withColumn('data_processed_at',
        when(col('data_processed_at').isNull(), lit(max_ts))
        .otherwise(col('data_processed_at')))
    
    # Create trimmed original timestamp column
    # Pandas: sc["data_processed_at_og_sc"] = pd.to_datetime(sc["data_processed_at"], errors="coerce").dt.floor("s")
    sc = sc.withColumn('data_processed_at_og_sc',
        date_trunc('second', col('data_processed_at')))
    
    print("data_processed_at_og_sc complete")
    
    # Bronze data_processed_at is preserved (no longer overwritten with ts_now())
    # The original value is also kept in data_processed_at_og_sc for reference
    
    # Add load_source_codes as source
    # Pandas: sc["source"] = "load_source_codes"
    sc = sc.withColumn('source', lit('load_source_codes'))
    
    # Update source if Source Code starts with 'BUDGET_SC'
    if 'SourceCode' in sc.columns:
        # Pandas: mask_budg_sc = sc["SourceCode"].astype(str).str.startswith("BUDGET_SC", na=False)
        # Pandas: sc.loc[mask_budg_sc, "source"] = "budget_missing_sc"
        sc = sc.withColumn('source',
            when(col('SourceCode').cast(StringType()).startswith('BUDGET_SC'),
                 lit('budget_missing_sc'))
            .otherwise(col('source')))
    
    ### Load
    # Create dataframe with standard schema
    # Pandas: sc_standard = pd.DataFrame(data=None,columns=sc_dims)
    # Pandas: sc_standard = sc_standard.astype(sc_def)
    sc_standard_schema = create_schema_from_dict(sc_dims, sc_def)
    sc_standard = spark.createDataFrame([], sc_standard_schema)
    
    # OPTION 1: THE EXPLICITLY INPUT SOURCECODES
    # Pandas: insert_df = sc.copy()
    insert_df = sc  # PySpark DataFrames are immutable
    
    # Pandas: insert_df = database_headers(insert_df)
    insert_df = database_headers_pyspark(insert_df)
    
    # Build client_campaign column (complex logic from lines 283-291)
    insert_df = insert_df.withColumn('client_campaign',
        concat(
            upper(col('client')),
            lit('-'),
            coalesce(col('campaign_code'), lit('')),
            when(col('campaign_name').isNotNull(),
                 concat(lit('-'), col('campaign_name')))
            .otherwise(lit(''))
        ))
    
    # Remove trailing hyphen
    insert_df = insert_df.withColumn('client_campaign',
        regexp_replace(col('client_campaign'), '-$', ''))
    
    # If client_campaign is just the client name, add source_code
    insert_df = insert_df.withColumn('client_campaign',
        when(col('client_campaign') == upper(col('client')),
             concat(col('client_campaign'), lit('-'), col('source_code')))
        .otherwise(col('client_campaign')))
    
    # Append -FY{fy} if fy is available
    # Pandas: insert_df['client_campaign'] += insert_df['fy'].apply(lambda x: f"-MD_FY{x}" if pd.notna(x) else "")
    insert_df = insert_df.withColumn('client_campaign',
        when(col('fy').isNotNull(),
             concat(col('client_campaign'), lit('-MD_FY'), col('fy').cast(StringType())))
        .otherwise(col('client_campaign')))
    
    # Add source_code_key using dense_rank (equivalent to pd.factorize)
    # Pandas: insert_df['source_code_key'], _ = pd.factorize(insert_df['source_code'])
    window_spec = Window.orderBy('source_code')
    insert_df = insert_df.withColumn('source_code_key',
                                     (dense_rank().over(window_spec) - 1).cast(LongType()))
    
    # Select overlap columns and union with sc_standard
    # Pandas: overlap_cols = list(set(insert_df.columns) & set(sc_dims))
    # Pandas: sc_standard = pd.concat([sc_standard,insert_df[overlap_cols]])
    overlap_cols = list(set(insert_df.columns) & set(sc_dims))
    sc_standard = sc_standard.unionByName(insert_df.select(*overlap_cols), allowMissingColumns=True)
    
    # Add curated_sc_key
    sc_standard = sc_standard.withColumn('sc_key_curated',
        concat(
            coalesce(upper(col('client')), lit('')),
            lit('_sc_'),
            coalesce(col('source_code').cast(StringType()), lit(''))
        ))
    
    # OPTION 3: UNSOURCED FALLBACKS
    # Generate years DataFrame
    # Pandas: years = pd.DataFrame(range(min_gift.year-1,max_gift.year+1), columns=['fy'])
    # Pandas: years['fy']=years['fy'].astype(str)
    # Pandas: blank = pd.DataFrame(data=['XXXX'],columns=['fy'])
    # Pandas: years = pd.concat([years,blank])
    years_data = [(str(y),) for y in range(min_gift.year - 1, max_gift.year + 1)]
    years_data.append(('XXXX',))
    years = spark.createDataFrame(years_data, ['fy'])
    
    # Add columns to years
    # Pandas: years['client']=client
    years = years.withColumn('client', lit(client))
    # Pandas: years['source_code']='UNSOURCED-FY'+years['fy']
    years = years.withColumn('source_code', concat(lit('UNSOURCED-FY'), col('fy')))
    # Pandas: years['campaign_code']='UNSOURCED-FY'+years['fy']
    years = years.withColumn('campaign_code', concat(lit('UNSOURCED-FY'), col('fy')))
    # Pandas: years['campaign_name']='Unsourced FY'+years['fy']
    years = years.withColumn('campaign_name', concat(lit('Unsourced FY'), col('fy')))
    # Pandas: years['client_campaign']=years['client'].str.upper()+'-UNSOURCED-FY'+years['fy']
    years = years.withColumn('client_campaign',
        concat(upper(col('client')), lit('-UNSOURCED-FY'), col('fy')))
    
    # Add source_code_key using factorize equivalent
    # Pandas: years['source_code_key'], _ = pd.factorize(years['client_campaign'])
    window_spec = Window.orderBy('client_campaign')
    years = years.withColumn('source_code_key', (dense_rank().over(window_spec) - 1).cast(LongType()))
    
    # Add mail_date (only for non-XXXX rows)
    # Pandas: years['mail_date']=years[years['fy']!='XXXX'].apply(lambda x: date(int(x['fy']),schema['firstMonthFiscalYear'],1), axis=1)
    years = years.withColumn('mail_date',
        when(col('fy') != 'XXXX',
             to_timestamp(concat(
                col('fy'),
                lit('-'),
                lit(str(schema['firstMonthFiscalYear']).zfill(2)),
                lit('-01')
             )).cast(TimestampNTZType()))
        .otherwise(lit(None).cast(TimestampNTZType())))
    
    # Set XXXX back to null
    # Pandas: years['fy']=years['fy'].replace('XXXX', None)
    years = years.withColumn('fy',
        when(col('fy') == 'XXXX', lit(None).cast(LongType()))
        .otherwise(col('fy').try_cast(LongType())))
    
    # Pandas: years['source']='unsourced fallbacks'
    years = years.withColumn('source', lit('unsourced fallbacks'))
    
    # Add curated_sc_key
    years = years.withColumn('sc_key_curated',
        concat(
            coalesce(upper(col('client')), lit('')),
            lit('_unsourced_FY'),
            coalesce(col('fy').cast(StringType()), lit(''))
        ))
    
    # Union with sc_standard
    # Pandas: overlap=list(set(years.columns) & set(sc_dims))
    # Pandas: sc_standard = pd.concat([sc_standard,years[overlap]])
    overlap = list(set(years.columns) & set(sc_dims))
    sc_standard = sc_standard.unionByName(years.select(*overlap), allowMissingColumns=True)
    
    # Cast date columns to timestamp (mail_date → NTZ, mail_date_original → LTZ per schema)
    if 'mail_date' in sc_standard.columns:
        # Pandas: sc_standard['mail_date'] = pd.to_datetime(sc_standard['mail_date'], errors='coerce')
        sc_standard = sc_standard.withColumn('mail_date', 
            coalesce(to_timestamp(col('mail_date')), lit(None)).cast(TimestampNTZType()))
    
    if 'mail_date_original' in sc_standard.columns:
        # Pandas: sc_standard['mail_date_original'] = pd.to_datetime(sc_standard['mail_date_original'], errors='coerce')
        sc_standard = sc_standard.withColumn('mail_date_original', 
            coalesce(to_timestamp(col('mail_date_original')), lit(None)).cast(TimestampType()))
    
    # Cast integer columns
    for _icol in ('fy', 'fy_mail_date_original'):
        if _icol in sc_standard.columns:
            # Pandas: sc_standard[_icol] = pd.to_numeric(sc_standard[_icol], errors='coerce').astype('Int64')
            sc_standard = sc_standard.withColumn(_icol, col(_icol).try_cast(LongType()))
    
    # Cast fiscal year columns to StringType (intentional business logic)
    # These are stored as strings in Silver even though Bronze has them as long
    for _fycol in ('fy_sc', 'fy_month_sc', 'fy_quarter_sc'):
        if _fycol in sc_standard.columns:
            sc_standard = sc_standard.withColumn(_fycol, trim(col(_fycol).cast(StringType())))
    
    # Import (overwrite replaces data)
    table_name = 'source_code'
    
    # Preserve bronze data_processed_at where available; use current timestamp for synthetic rows (unsourced/missing)
    sc_standard = sc_standard.withColumn('data_processed_at',
        coalesce(col('data_processed_at'), lit(ts_now()).cast(TimestampType())))
    
    # PERF: Write deferred — sc_standard is combined with missing SCs in sc_find_missing_pyspark
    etl2_status_entry(client, 'Source Code Processing: Loaded')
    
    return sc, sc_standard

# COMMAND ----------

# DBTITLE 1,def sc_find_missing_pyspark
def sc_find_missing_pyspark(df, sc, sc_standard=None):
    """
    PySpark version - Original lines 388-490
    Find campaigns that aren't in SourceCode.csv already
    """
    # Pandas: cols = df.columns[~df.columns.str.upper().str.contains(client.upper())]
    # Note: This line references 'client' variable from outer scope
    cols = [c for c in df.columns if client.upper() not in c.upper()]
    
    # Pandas: trx = database_headers(df)
    trx = database_headers_pyspark(df)
    # Pandas: sc2 = database_headers(sc)
    sc2 = database_headers_pyspark(sc)
    
    # Pandas: comb = pd.merge(trx, sc2, on='source_code', how='left', suffixes=['','_sc'], indicator=True)
    # Pandas: missing = comb[comb['_merge'] != 'both'].copy()
    # PySpark: Use left_anti join to find rows in trx NOT in sc2
    # This avoids column ambiguity completely
    missing = trx.join(sc2.select('source_code'), on='source_code', how='left_anti')
    
    # Fix nulls — replace empty/'NONE' with NULL but keep NULL rows
    # Pandas: missing['source_code'] = missing['source_code'].replace(['', 'NONE'], np.nan)
    # Pandas: missing = missing[missing['source_code'] != '']
    # Note: In pandas NaN != '' → True, so NaN rows are KEPT. The filter is a no-op after replace.
    # PySpark fix: only replace, do NOT filter isNotNull() — that drops valid missing SC records.
    missing = missing.withColumn('source_code',
        when((col('source_code') == '') | (col('source_code') == 'NONE'), lit(None))
        .otherwise(col('source_code')))
    
    # Pandas: missing['source'] = 'inferred from trx'
    missing = missing.withColumn('source', lit('inferred from trx'))
    # Pandas: missing['data_processed_at'] = ts_now()
    missing = missing.withColumn('data_processed_at', lit(ts_now()).cast(TimestampType()))
    
    # Estimate mail date
    # Pandas: missing['tmp'] = missing['source_code'].fillna(missing['campaign_name'])
    # No lit('') fallback: when both source_code and campaign_name are NULL,
    # coalesce returns NULL → concat(NULL, '_', fy) → NULL → single partition.
    # This matches pandas where NaN + '_' + str(fy) = NaN → one group.
    missing = missing.withColumn('tmp', coalesce(col('source_code'), col('campaign_name')))
    
    # Add gift fy to tmp for grouping
    # Pandas: missing['tmp'] = missing['tmp'] + "_" + fiscal_from_column(missing, 'gift_date', schema['firstMonthFiscalYear']).astype(str)
    fy_expr = fiscal_from_column_pyspark(missing, 'gift_date', schema['firstMonthFiscalYear'])
    missing = missing.withColumn('tmp',
        concat(col('tmp'), lit('_'), fy_expr.cast(StringType())))
    
    # Group by tmp to get min mail_date
    # Pandas: missing['mail_date'] = missing.groupby('tmp')['gift_date'].transform('min')
    window_spec = Window.partitionBy('tmp')
    missing = missing.withColumn('mail_date',
        spark_min('gift_date').over(window_spec))
    
    # Add fy for missing dates grouped
    # Pandas: missing['fy'] = fiscal_from_column(missing, 'mail_date', schema['firstMonthFiscalYear'])
    missing = missing.withColumn('fy',
        fiscal_from_column_pyspark(missing, 'mail_date', schema['firstMonthFiscalYear']).try_cast(LongType()))
    
    # Build client_campaign column (complex apply logic from lines 415-424)
    # Pandas: missing['client_campaign'] = missing.apply(lambda row: ...)
    missing = missing.withColumn('client_campaign',
        # If campaign_code AND campaign_name present
        when(col('campaign_code').isNotNull() & col('campaign_name').isNotNull(),
             concat(upper(col('client')), lit('-'), col('campaign_code'), lit('-'), col('campaign_name')))
        # Elif campaign_code present
        .when(col('campaign_code').isNotNull(),
              concat(upper(col('client')), lit('-'), col('campaign_code')))
        # Elif campaign_name present
        .when(col('campaign_name').isNotNull(),
              concat(upper(col('client')), lit('-CN '), col('campaign_name')))
        # Elif source_code present
        .when(col('source_code').isNotNull(),
              concat(upper(col('client')), lit('-'), col('source_code')))
        # Else just client
        .otherwise(upper(col('client'))))
    
    # Append -FY{fy} if fy is available
    missing = missing.withColumn('client_campaign',
        when(col('fy').isNotNull(),
             concat(col('client_campaign'), lit('-MD_FY'), col('fy').cast(StringType())))
        .otherwise(col('client_campaign')))
    
    # Narrow missing to cols in sc_dims
    # Pandas: to_use = list(set(missing.columns) & set(sc_dims))
    # Pandas: missing = missing[to_use].drop_duplicates().reset_index(drop=True)
    to_use = list(set(missing.columns) & set(sc_dims))
    missing = missing.select(*to_use).dropDuplicates()
    
    # Note: PySpark doesn't track duplicate column names at runtime
    # Duplicate columns would cause error during select
    print("[OK] No duplicate columns in PySpark DataFrame structure")
    
    # Backfill campaign_group in `missing` from `sc2` using campaign_code
    if 'campaign_group' not in missing.columns:
        missing = missing.withColumn('campaign_group', lit(None).cast(StringType()))
    
    # Normalize blanks -> NA
    # Pandas: missing['campaign_group'] = missing['campaign_group'].replace(['', ' ', 'NONE', 'None'], np.nan)
    missing = missing.withColumn('campaign_group',
        when(col('campaign_group').isin('', ' ', 'NONE', 'None'), lit(None))
        .otherwise(col('campaign_group')))
    missing = missing.withColumn('campaign_code',
        when(col('campaign_code').isin('', ' ', 'NONE', 'None'), lit(None))
        .otherwise(col('campaign_code')))
    
    # Same for sc2
    if 'campaign_group' not in sc2.columns:
        sc2 = sc2.withColumn('campaign_group', lit(None).cast(StringType()))
    
    sc2 = sc2.withColumn('campaign_group',
        when(col('campaign_group').isin('', ' ', 'NONE', 'None'), lit(None))
        .otherwise(col('campaign_group')))
    sc2 = sc2.withColumn('campaign_code',
        when(col('campaign_code').isin('', ' ', 'NONE', 'None'), lit(None))
        .otherwise(col('campaign_code')))
    
    # Build lookup: campaign_code -> campaign_group
    # Pandas: sc2_lookup = sc2.loc[sc2['campaign_code'].notna() & sc2['campaign_group'].notna(), ['campaign_code', 'campaign_group']].drop_duplicates()
    sc2_lookup = sc2.filter(
        col('campaign_code').isNotNull() & col('campaign_group').isNotNull()
    ).select('campaign_code', 'campaign_group').dropDuplicates()
    
    # If duplicates exist, take the most frequent (mode)
    # Pandas: sc2_lookup.groupby('campaign_code')['campaign_group'].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    # PySpark: Use count and pick most common, or first if tie
    # Window already imported at module level
    window = Window.partitionBy('campaign_code').orderBy(F.desc('cnt'))
    sc2_lookup = (sc2_lookup
                  .groupBy('campaign_code', 'campaign_group')
                  .agg(spark_count('*').alias('cnt'))
                  .withColumn('rn', row_number().over(window))
                  .filter(col('rn') == 1)
                  .select('campaign_code', 'campaign_group'))
    
    # Map onto missing
    # Pandas: missing['campaign_group'] = missing['campaign_group'].fillna(missing['campaign_code'].map(sc2_lookup.set_index('campaign_code')['campaign_group']))
    missing = missing.join(sc2_lookup.withColumnRenamed('campaign_group', 'campaign_group_lookup'),
                          on='campaign_code', how='left')
    missing = missing.withColumn('campaign_group',
        coalesce(col('campaign_group'), col('campaign_group_lookup')))
    missing = missing.drop('campaign_group_lookup')
    
    _cg_stats = missing.agg(
        spark_sum(when(col('campaign_group').isNotNull(), lit(1)).otherwise(lit(0))).alias('_cg'),
        spark_count(lit(1)).alias('_tot'),
    ).first()
    print(
        f"[BACKFILL] Backfilled campaign_group for {int(_cg_stats['_cg'] or 0):,} / "
        f"{int(_cg_stats['_tot'] or 0):,} rows (non-null)"
    )
    
    # Add identifiers
    # Pandas: missing['source_code_key'] = [str(i) for i in range(1, len(missing) + 1)]
    missing = missing.withColumn('source_code_key',
        (row_number().over(Window.orderBy(monotonically_increasing_id()))).cast(LongType()))
    
    # Load missing source codes separately
    table_name = 'source_code_missing'
    # Set audit timestamps for tracking when the table was loaded
    missing = missing.withColumn('ingest_ts', current_timestamp())
    missing = missing.withColumn('update_ts', current_timestamp())
    sql_import_pyspark(missing, table_name, overwrite_or_append='overwrite')
    
    missing = apply_client_source_code_missing_transform(missing, client)
    
    # Add curated_sc_key
    missing = missing.withColumn('sc_key_curated',
        concat(
            coalesce(upper(col('client')), lit('')),
            lit('_trx_'),
            coalesce(col('source_code').cast(StringType()), lit(''))
        ))
    
    # OPTION 2: INFERRED FROM TRX
    # OPTIMIZED: Combined write with sc_standard (replaces separate overwrite + delete + append)
    table_name = 'source_code'
    if sc_standard is not None:
        # --- DIAGNOSTIC LOGGING (BUG 12) — gated: only runs at DEBUG level ---
        if logger.isEnabledFor(logging.DEBUG):
            from pyspark.sql.types import StringType as _ST, LongType as _LT, IntegerType as _IT
            _std_fields = {f.name: f.dataType for f in sc_standard.schema.fields}
            _mis_fields = {f.name: f.dataType for f in missing.schema.fields}
            _common = set(_std_fields.keys()) & set(_mis_fields.keys())
            _mismatches = [(c, str(_std_fields[c]), str(_mis_fields[c])) for c in sorted(_common) if _std_fields[c] != _mis_fields[c]]
            if _mismatches:
                logger.debug("[DIAG] TYPE MISMATCHES between sc_standard and missing:")
                for _c, _t1, _t2 in _mismatches:
                    logger.debug(f"  {_c}: sc_standard={_t1}  vs  missing={_t2}")
            else:
                logger.debug("[DIAG] No type mismatches on common columns")
            _str_cols_std = [f.name for f in sc_standard.schema.fields if isinstance(f.dataType, _ST)]
            if _str_cols_std:
                _na_expr = [spark_sum(when(col(c) == '<NA>', lit(1)).otherwise(lit(0))).alias(c) for c in _str_cols_std]
                _na_row = sc_standard.select(_na_expr).first()
                _na_cols = [c for c in _str_cols_std if (_na_row[c] or 0) > 0]
                if _na_cols:
                    logger.debug(f"[DIAG] <NA> in sc_standard STRING cols: {_na_cols}")
                    for _nc in _na_cols:
                        logger.debug(f"  {_nc}: type_in_missing={_mis_fields.get(_nc, 'NOT_PRESENT')}")
            _str_cols_mis = [f.name for f in missing.schema.fields if isinstance(f.dataType, _ST)]
            if _str_cols_mis:
                _na_expr_m = [spark_sum(when(col(c) == '<NA>', lit(1)).otherwise(lit(0))).alias(c) for c in _str_cols_mis]
                _na_row_m = missing.select(_na_expr_m).first()
                _na_cols_m = [c for c in _str_cols_mis if (_na_row_m[c] or 0) > 0]
                if _na_cols_m:
                    logger.debug(f"[DIAG] <NA> in missing STRING cols: {_na_cols_m}")
                    for _nc in _na_cols_m:
                        logger.debug(f"  {_nc}: type_in_sc_standard={_std_fields.get(_nc, 'NOT_PRESENT')}")
            _only_std = set(_std_fields.keys()) - set(_mis_fields.keys())
            _only_mis = set(_mis_fields.keys()) - set(_std_fields.keys())
            _str_only_std = [c for c in _only_std if isinstance(_std_fields[c], _ST)]
            _str_only_mis = [c for c in _only_mis if isinstance(_mis_fields[c], _ST)]
            if _str_only_std or _str_only_mis:
                logger.debug(f"[DIAG] STRING cols only in sc_standard: {sorted(_str_only_std)}")
                logger.debug(f"[DIAG] STRING cols only in missing: {sorted(_str_only_mis)}")
            logger.debug("[DIAG] --- END ---")


        combined_sc = sc_standard.unionByName(missing, allowMissingColumns=True)
        # Set audit timestamps for tracking when the table was loaded
        combined_sc = combined_sc.withColumn('ingest_ts', current_timestamp())
        combined_sc = combined_sc.withColumn('update_ts', current_timestamp())
        sql_import_pyspark(combined_sc, table_name, overwrite_or_append='overwrite')
    else:
        # Legacy path: delete inferred rows and append
        full_table_path = get_dbo_table_path(table_name)
        script = f"delete from {full_table_path} where source = 'inferred from trx'"
        sql_exec_only(script)
        sql_import_pyspark(missing, table_name, overwrite_or_append='append')
    
    missing_count = int(missing.agg(spark_count(lit(1)).alias('_mc')).first()['_mc'] or 0)
    etl2_status_entry(
        client,
        f'Source Code Processing: {missing_count} sources in tx data not in SourceCode.csv',
    )

# COMMAND ----------

# DBTITLE 1,def qa_source_codes (no changes - uses SQL)
def _qa_sc_unexpected_blanks(client):
    """No changes - already uses SQL queries"""
    table_name = 'source_code'
    full_table_path = get_dbo_table_path(table_name)
    query = f"select count(distinct source_code) unexpectedly_empty_scs from {full_table_path} where source_code <> '' and source <> 'inferred from trx'"
    sdf = _get_spark().sql(query)
    row = sdf.first() if sdf is not None else None
    results = int(row[0]) if row is not None else 0
    add_to_error_table(client,
                       'Source Codes populated',
                       'Source Code',
                       f'{results} source codes processed',
                       results == 0,
                       'error')

def _qa_sc_placeholders(client):
    """No changes - already uses SQL queries"""
    table_name = 'source_code'
    full_table_path = get_dbo_table_path(table_name)
    query = f"select count(distinct source_code) placeholders_inserted from {full_table_path} where source = 'unsourced fallbacks'"
    sdf = _get_spark().sql(query)
    row = sdf.first() if sdf is not None else None
    results = int(row[0]) if row is not None else 0
    add_to_error_table(client,
                       'Placeholders populated',
                       'Source Code',
                       f'{results} placeholders for unsourced gifts inserted',
                       results == 0,
                       'error')

def qa_source_codes(client):
    """No changes - orchestration only"""
    _qa_sc_unexpected_blanks(client)
    _qa_sc_placeholders(client)
    
    warnings, errors = query_qa_errors(client, ['Source Code'])
    etl2_status_entry(client, f'Source Code Processing: QAd (Warnings: {warnings}, Errors: {errors})')
    
    if apply_client_force_publish_source_code_despite_errors(client):
        errors = 0
    if errors == 0:
        print("No errors found. Publishing to Curated")
        publish_to_curated('source_code', False)
        etl2_status_entry(client, 'Source Code Processing: Published to Curated')

# COMMAND ----------

# DBTITLE 1,def budget_load_and_process_csv_pyspark
def budget_load_and_process_csv_pyspark(client):
    """
    PySpark version - Original lines 554-623
    Load and process budget data
    """
    schema = get_schema(client)
    
    ################################################
    # Read Budget from bronze Delta table
    ################################################
    df = load_budget_pyspark(client)
    if df is None:
        raise Exception(f"Failed to load Budget for client {client}")
    
    # Pandas: df.columns = df.columns.str.strip()
    for old_col in df.columns:
        new_col = old_col.strip()
        if old_col != new_col:
            df = df.withColumnRenamed(old_col, new_col)
    
    # Standardize column names
    # Pandas: df.rename(columns={'Data_Processed_At': 'data_processed_at'}, inplace=True)
    if 'Data_Processed_At' in df.columns:
        df = df.withColumnRenamed('Data_Processed_At', 'data_processed_at')
    
    ################################################
    # Cast columns based on Budget schema (PySpark types)
    ################################################
    # Load schema from JSON file for single source of truth
    budget_schema = get_table_schemas()['BudgetTable']
    
    # Map Bronze column names to Silver schema column names
    # Bronze uses CamelCase, Silver uses snake_case
    bronze_to_silver_mapping = {
        'Client': 'client',
        'Budg_Type': 'budg_type',
        'Budg_FY': 'budg_fy',
        'Budg_FYQtr': 'budg_fy_qtr',
        'Budg_FYMonth': 'budg_fy_month',
        'Budg_CampaignCode': 'budg_campaign_code',
        'Budg_CampaignName': 'budg_campaign_name',
        'Budg_CampaignGroup': 'budg_campaign_group',
        'Budg_Channel': 'budg_channel',
        'Budg_Program': 'budg_program',
        'Budg_MailDate': 'budg_mail_date',
        'Budg_Quantity': 'budg_quantity',
        'Budg_Gifts': 'budg_gifts',
        'Budg_Revenue': 'budg_revenue',
        'Budg_Cost': 'budg_cost',
        'Budg_YE_Gifts': 'budg_ye_gifts',
        'Budg_YE_Revenue': 'budg_ye_revenue',
        'Budg_Client_CampCode': 'budg_client_campcode',
        'Data_Processed_At': 'data_processed_at'
    }
    
    # Cast each column based on its target schema type
    for bronze_col, silver_col in bronze_to_silver_mapping.items():
        if bronze_col in df.columns and silver_col in budget_schema:
            pyspark_type = budget_schema[silver_col]
            
            if pyspark_type == 'TimestampType':
                # Already cast to TimestampType in load_budget_pyspark - no additional casting needed
                pass
            elif pyspark_type == 'DoubleType':
                # Cast to double with error handling (coerces invalid values to null)
                df = df.withColumn(bronze_col, col(bronze_col).cast(DoubleType()))
            elif pyspark_type == 'StringType':
                # Cast to string and trim whitespace
                df = df.withColumn(bronze_col, trim(col(bronze_col).cast(StringType())))
            elif pyspark_type == 'LongType':
                # Cast to long integer
                df = df.withColumn(bronze_col, col(bronze_col).cast(LongType()))
    
    # Note: PySpark handles nulls natively, no need for pd.notnull replacement
    
    ################################################
    # Return cleaned DataFrame
    ################################################
    col_count = len(df.columns)
    print(f"[OK] Budget DataFrame ready with {col_count} columns.")
    return df

# COMMAND ----------

# DBTITLE 1,def process_budget_table_pyspark
def process_budget_table_pyspark(budget_df, client):
    """
    PySpark version - Process and load budget table.
    REFACTORED: Uses enforce_full_json_schema. Always full overwrite.
    """
    logger.warning('****** Budget Table Load Started ******')
    
    # Setup
    table_name = 'budget'
    etl2_status_entry(client, 'BI Table Processing: Budget Table Started')
    
    # Prepare DataFrame
    # Ensure data_processed_at column exists (some clients' bronze tables don't have it)
    if 'data_processed_at' not in budget_df.columns:
        budget_df = budget_df.withColumn('data_processed_at', lit(None).cast(TimestampType()))
    # Preserve original bronze data_processed_at; fall back to current timestamp for nulls
    budget_df = budget_df.withColumn('data_processed_at',
        coalesce(col('data_processed_at'), lit(ts_now()).cast(TimestampType())))
    
    # Align column names to match SQL naming
    budget_df_db = database_headers_pyspark(budget_df)
    
    # Enforce JSON schema (strict: only JSON columns, correct types)
    budget_json_schema = _load_schema_from_json('budget')
    if budget_json_schema:
        budget_df_db = enforce_full_json_schema(budget_df_db, budget_json_schema)
    else:
        # Fallback: keep only columns defined in table_schema['BudgetTable']
        valid_cols = table_schema['BudgetTable']
        existing_valid_cols = [c for c in budget_df_db.columns if c in valid_cols]
        budget_df_db = budget_df_db.select(*existing_valid_cols)
    
    # Set audit timestamps for tracking when the table was loaded
    budget_df_db = budget_df_db.withColumn('ingest_ts', current_timestamp())
    budget_df_db = budget_df_db.withColumn('update_ts', current_timestamp())
    
    import time
    start_time = time.time()
    logger.warning(f"Inserting rows into {table_name}")
    
    sql_import_pyspark(budget_df_db, table_name, overwrite_or_append='overwrite')
    
    elapsed = time.time() - start_time
    logger.warning(f"****** Budget Load Complete in {elapsed:.2f}s ******")

# COMMAND ----------

# DBTITLE 1,def process_trx_apply_suppressions_pyspark
def process_trx_apply_suppressions_pyspark(client):
    """
    PySpark version - Original lines 686-728
    Load data and apply suppressions.
    Uses incremental read when get_table_mode('gift') == 'incremental'.
    """
    # Determine if incremental read is enabled for gift
    last_wm = None
    if is_incremental_enabled() and get_table_mode('gift') == 'incremental':
        last_wm, _ = get_watermark(_get_spark(), client, 'gift', 'dbo_gift')
        if last_wm is not None:
            logger.info(f'[INCREMENTAL READ] gift watermark for {client}: {last_wm}')
        else:
            logger.info(f'[INCREMENTAL READ] No watermark found for {client} \u2014 full read')
    
    # Load data.parquet (incremental or full based on watermark)
    df = load_parquet_pyspark(client, last_watermark=last_wm)
    
    # Suppressions
    schema = get_schema(client)
    suppressions = schema['Suppressions']
    for k in suppressions.keys():
        df = apply_func(df, suppressions, k, client=client)
    logger.info(f'Suppressions applied: {list(suppressions.keys())}')
    
    # If multiple gift ids (AFHU)
    if ('GiftID' in df.columns) and ('Gift_ID' in df.columns):
        df = df.withColumn('GiftID',
            when((col('GiftID').isNull()) | (col('GiftID') == 'nan'),
                 col('Gift_ID'))
            .otherwise(col('GiftID')))
        df = df.drop('Gift_ID')
    
    # General Additions
    df = df.withColumn('GiftFY',
        fiscal_from_column_pyspark(df, 'GiftDate', schema['firstMonthFiscalYear']).cast(IntegerType()))
    
    return df

# COMMAND ----------

# DBTITLE 1,def process_trx_add_fh_filters_pyspark
def process_trx_add_fh_filters_pyspark(df, client):
    """
    PySpark version - Original lines 735-762
    Add File Health filters
    """
    try:
        # NOTE: 'schema' must be global so parser filter functions
        # (JoinYear, JoinFiscal, JoinFiscalYear) can access it
        global schema
        schema = get_schema(client)
        filters = get_fh_filters(schema['FileHealth'])

        # Add filters that all clients should have
        generics = ['DonorGroup', 'GiftLevel', 'GiftMonth', 'GiftHistory', 'GiftAmountFlag', 'JoinLevel']
        filters.extend([x for x in generics if x not in filters])

        # Remove some filters that will now be handled elsewhere
        removes = ['GiftFiscal', 'JoinFiscal', 'JoinFiscalYear', 'JoinFY', 'GiftFY']
        filters = [x for x in filters if x not in removes]

        # PERF: FirstGiftDate + DonorGroup + GiftHistory each did groupBy(DonorID)+join.
        # One aggregate + one join matches the same columns (incl. synthetic GiftID rule).
        have_dg = 'DonorGroup' in filters
        have_gh = 'GiftHistory' in filters
        if have_dg and have_gh:
            if 'GiftID' not in df.columns:
                df = df.withColumn('GiftID', monotonically_increasing_id())
            # Exclude epoch artifact rows (1970-01-01) from donor stats.
            # These are NULL dates cast to epoch during bronze ingestion and get
            # filtered later in process_trx_prep_data_pyspark. Including them here
            # inflates DonorGroup/GiftHistory to 'Multi' for single-gift donors.
            _not_epoch = col('GiftDate').cast('date') != lit('1970-01-01').cast('date')
            _donor_stats = df.groupBy('DonorID').agg(
                spark_min(when(_not_epoch, col('GiftDate'))).alias('FirstGiftDate'),
                spark_sum(when(_not_epoch, lit(1)).otherwise(lit(0))).alias('_dg_n'),
                countDistinct(when(_not_epoch, col('GiftID'))).alias('_gh_n'),
            )
            _donor_stats = (
                _donor_stats.withColumn(
                    'DonorGroup',
                    when(col('_dg_n') > 1, lit('Multi')).otherwise(lit('Single')),
                )
                .withColumn(
                    'GiftHistory',
                    when(col('_gh_n') > 1, lit('Multi')).otherwise(lit('Single')),
                )
                .drop('_dg_n', '_gh_n')
            )
            # Pandas: df = df.merge(_donor_stats, on='DonorID', how='left')
            # PySpark: Ensure no column ambiguity by selecting only needed columns from _donor_stats
            _donor_stats_cols = ['DonorID', 'FirstGiftDate', 'DonorGroup', 'GiftHistory']
            _donor_stats_select = _donor_stats.select([c for c in _donor_stats_cols if c in _donor_stats.columns])
            df = df.join(_donor_stats_select, on='DonorID', how='left')
            filters = [f for f in filters if f not in ('DonorGroup', 'GiftHistory')]
        else:
            df = get_fh_FirstGiftDate(df)

        # Pandas: df['gift_id_bkup']=df['GiftID']
        df = df.withColumn('gift_id_bkup', col('GiftID'))

        for f in filters:
            df = apply_filters(df, f)

        return df
    
    except Exception as e:
        logger.error("process_trx_add_fh_filters: %s", traceback.format_exc())
        notification_exception_raised(client, code_location='process_trx_add_fh_filters', error_message=repr(e))
        raise(e)

# COMMAND ----------

# DBTITLE 1,def process_trx_add_cp_filters_pyspark
def process_trx_add_cp_filters_pyspark(df, client):
    """
    PySpark version - Original lines 771-865
    Add campaign performance filters
    """
    ### Load source code
    # Get min/max gift dates in a single action (not two separate collects)
    date_range = df.agg(spark_max('GiftDate').alias('max_dt'), spark_min('GiftDate').alias('min_dt')).first()
    max_gift = date_range['max_dt']
    min_gift = date_range['min_dt']
    
    # Fallback: when incremental batch is empty (all GiftDate NULL),
    # read date range from existing gift silver table so source code
    # fiscal-year range generation still works.
    if min_gift is None or max_gift is None:
        _gift_dbo = get_dbo_table_path('gift')
        if spark.catalog.tableExists(_gift_dbo):
            _existing = spark.table(_gift_dbo).agg(
                spark_max('gift_date').alias('mx'),
                spark_min('gift_date').alias('mn')
            ).first()
            max_gift = max_gift or _existing['mx']
            min_gift = min_gift or _existing['mn']
        # Last resort: if gift silver is also empty, use today
        if min_gift is None or max_gift is None:
            from datetime import date as _date
            min_gift = min_gift or _date.today()
            max_gift = max_gift or _date.today()
        logger.warning(f'GiftDate batch range was None \u2014 using fallback: min={min_gift}, max={max_gift}')
    
    sc, sc_standard = sc_load_and_process_csv_pyspark(client, max_gift=max_gift, min_gift=min_gift)
    
    # Create white mail source codes
    schema = get_schema(client)
    if 'SynthSC' in schema.keys():
        df = add_synth_sc(df, schema)
    
    # Save appeal to campaign mapping (SC data is small \u2014 collect is fine)
    _sc = sc.dropDuplicates(['CampaignCode']).select('CampaignCode', 'CampaignName')
    appeals_to_campaigns_rows = _sc.collect()
    appealsToCampaigns = {row['CampaignCode']: row['CampaignName'] for row in appeals_to_campaigns_rows if row['CampaignCode'] is not None}
    
    # Create broadcast map for join (small lookup)
    # Handle empty map case to avoid VOID type issues (test mode with all-NULL CampaignCodes)
    if len(appealsToCampaigns) > 0:
        appeals_map_expr = create_map([lit(x) for kv in appealsToCampaigns.items() for x in kv])
    else:
        # Create empty STRING->STRING map with dummy entry to establish type, then remove it
        appeals_map_expr = create_map([lit("__DUMMY__"), lit("__DUMMY__")])
        print("[WARNING] No valid CampaignCode mappings found - map lookup will return NULL")
    
    # Manual col renames to prevent downstream duplications
    # Bronze converts spaces to underscores, so we handle underscore versions
    ren = {
        'Campaign_Name': 'CampaignName',
        'Source_Code': 'SourceCode',
        'Campaign_Code': 'CampaignCode'
    }
    for old_name, new_name in ren.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    
    # Add CampaignCode with NULL if not in columns (avoid lit('') for map lookup safety)
    if "CampaignCode" not in df.columns:
        df = df.withColumn("CampaignCode", lit(None).cast(StringType()))
    
    # Rename CampaignCode from df to df_CampaignCode
    df = df.withColumnRenamed('CampaignCode', 'df_CampaignCode')
    _gift_cols_before_join = set(df.columns)
    
    ### Join tx and sc data
    # Rename ALL sc columns except join key to prevent ambiguity
    sc_renamed = sc
    for c in sc.columns:
        if c != 'SourceCode':
            sc_renamed = sc_renamed.withColumnRenamed(c, c + '_sc')
    
    dfc = df.join(sc_renamed, on='SourceCode', how='full_outer')
    
    # Use SC ListCode if Trx ListCode is na
    # Handle ListCode from sc (renamed to ListCode_sc) and df (ListCode)
    if 'ListCode_sc' in dfc.columns and 'ListCode' in dfc.columns:
        dfc = dfc.withColumn('ListCode', coalesce(col('ListCode'), col('ListCode_sc')))
        dfc = dfc.drop('ListCode_sc')
    elif 'ListCode_sc' in dfc.columns:
        # Only sc has ListCode - rename it
        dfc = dfc.withColumnRenamed('ListCode_sc', 'ListCode')
    # If only df has ListCode, keep it as is
    
    #############################################
    # Attempt to map in a campaign name
    #############################################
    # Coalesce CampaignCode from both sources (df and sc)
    if 'CampaignCode_sc' in dfc.columns:
        dfc = dfc.withColumn('CampaignCode',
            coalesce(col('CampaignCode_sc'), col('df_CampaignCode')))
        dfc = dfc.drop('CampaignCode_sc')
    else:
        dfc = dfc.withColumn('CampaignCode', col('df_CampaignCode'))
    
    # Map CampaignCode -> CampaignName using small lookup
    # Use coalesce to safely handle NULL or missing keys (returns NULL instead of failing)
    dfc = dfc.withColumn('TempCampName',
        when(col('CampaignCode').isNotNull(), appeals_map_expr[col('CampaignCode')])
        .otherwise(lit(None)))
    
    # CampaignName resolution \u2014 matches old Pandas merge + np.where behavior:
    # In Pandas merge(suffixes=['','_sc']):
    #   - If BOTH df and sc have CampaignName: df's stays as CampaignName, sc's \u2192 CampaignName_sc
    #   - If ONLY sc has CampaignName: sc's stays as CampaignName (no suffix, no conflict)
    # Then: dfc.CampaignName = np.where(dfc.CampaignName.isna(), TempCampName, dfc.CampaignName)
    #   \u2192 Only fills NULLs, never overrides existing non-null values.
    #
    # In PySpark: sc columns are ALWAYS suffixed with _sc, so we handle both cases:
    if 'CampaignName' in dfc.columns:
        # Transaction HAS CampaignName (WFP, CARE) \u2014 keep it, only NULL-fill from lookup
        dfc = dfc.withColumn('CampaignName',
            coalesce(col('CampaignName'), col('TempCampName')))
    elif 'CampaignName_sc' in dfc.columns:
        # Transaction has NO CampaignName (AFHU, FS) \u2014 SC's name is the only source
        # (In Pandas this would be the unsuffixed column since no conflict exists)
        dfc = dfc.withColumn('CampaignName',
            coalesce(col('CampaignName_sc'), col('TempCampName')))
    else:
        dfc = dfc.withColumn('CampaignName', col('TempCampName'))
    
    dfc = dfc.drop('TempCampName', 'df_CampaignCode')
    if 'CampaignName_sc' in dfc.columns:
        dfc = dfc.drop('CampaignName_sc')
    
    # Restore non-conflicting SC columns \u2014 match pandas merge(suffixes=['','_sc']) behavior.
    # In pandas, non-conflicting SC columns keep their original names (no suffix).
    # Only truly conflicting columns (same name on both gift and SC side) get dropped.
    # NOTE: Use single select instead of iterative withColumnRenamed because PySpark's
    # withColumnRenamed is case-insensitive and causes cascading renames (e.g. renaming
    # 'fy_sc' also renames 'FY_SC' since they match case-insensitively).
    sc_suffixed_cols = [c for c in dfc.columns if c.endswith('_sc')]
    to_restore = []
    to_drop = []
    for c_sc in sc_suffixed_cols:
        base_name = c_sc[:-3]
        if base_name in _gift_cols_before_join:
            to_drop.append(c_sc)
        else:
            to_restore.append(c_sc)
    rename_map = {c: c[:-3] for c in to_restore}
    drop_set = set(to_drop)
    select_exprs = []
    for c in dfc.columns:
        if c in drop_set:
            continue
        elif c in rename_map:
            select_exprs.append(col(f'`{c}`').alias(rename_map[c]))
        else:
            select_exprs.append(col(f'`{c}`'))
    dfc = dfc.select(select_exprs)
    if to_restore:
        print(f"[RESTORE] Restored {len(to_restore)} non-conflicting SC columns")
    if to_drop:
        print(f"[CLEANUP] Dropped {len(to_drop)} conflicting _sc columns: {to_drop}")
    
    # Map GiftID -> CampaignName using a JOIN instead of collecting 3.76M rows
    if 'CampaignName' in df.columns:
        from pyspark.sql.functions import broadcast
        gift_campaigns = df.select('GiftID', col('CampaignName').alias('_GiftCampName')).dropDuplicates(['GiftID'])
        dfc = dfc.join(broadcast(gift_campaigns), on='GiftID', how='left')
        dfc = dfc.withColumn('CampaignName',
            coalesce(col('CampaignName'), col('_GiftCampName')))
        dfc = dfc.drop('_GiftCampName')
    
    # Apply client specific filter restrictions
    dfc = add_dataset_filters(dfc, client)
    
    # Rename
    if 'ListCPP' in dfc.columns:
        dfc = dfc.withColumnRenamed('ListCPP', '_ListCPP')
    if 'Quantity' in dfc.columns:
        dfc = dfc.withColumnRenamed('Quantity', 'RawQuantity')
    
    # Clean ListCode - remove trailing .0 if it exists
    if 'ListCode' in dfc.columns:
        dfc = dfc.withColumn("ListCode",
            regexp_replace(col("ListCode").cast(StringType()), r"\.0$", ""))
    
    #############################################
    # Append SCs from trx that aren't in the csv
    #############################################
    sc_find_missing_pyspark(dfc, sc, sc_standard)
    
    #############################################
    # QA: Source Codes
    #############################################
    qa_source_codes(client)
    
    return dfc

# COMMAND ----------

# DBTITLE 1,def trx_backfill_sc_key_curated_pyspark
def trx_backfill_sc_key_curated_pyspark(df, sc_curated):
    """
    PySpark version - Original lines 880-1007
    Backfill sc_key_curated using multiple matching strategies
    """
    print(f"\n[BACKFILL] Starting sc_key_curated backfill")
    
    # Pandas: out = df.copy().reset_index(drop=True)
    out = df  # PySpark DataFrames are immutable
    
    # Initialize columns
    out = out.withColumn("sc_key_curated", lit(None).cast(StringType()))
    out = out.withColumn("sc_key_match_method", lit(None).cast(StringType()))
    
    # --------------------
    # Normalize columns
    # --------------------
    print("[NORMALIZE] Normalizing join columns...")
    
    out = out.withColumn("_client",
        coalesce(trim(col("client").cast(StringType())), lit("")))
    
    # Handle optional columns
    if "source_code" in out.columns:
        out = out.withColumn("_source_code",
            coalesce(trim(col("source_code").cast(StringType())), lit("")))
    else:
        out = out.withColumn("_source_code", lit(""))
    
    if "campaign_name" in out.columns:
        out = out.withColumn("_camp_name",
            coalesce(trim(col("campaign_name").cast(StringType())), lit("")))
    else:
        out = out.withColumn("_camp_name", lit(""))
    
    if "campaign_code" in out.columns:
        out = out.withColumn("_camp_code",
            coalesce(trim(col("campaign_code").cast(StringType())), lit("")))
    else:
        out = out.withColumn("_camp_code", lit(""))
    
    # Same for sc_curated
    sc = sc_curated
    sc = sc.withColumn("_client",
        coalesce(trim(col("client").cast(StringType())), lit("")))
    
    if "source_code" in sc.columns:
        sc = sc.withColumn("_source_code",
            coalesce(trim(col("source_code").cast(StringType())), lit("")))
    else:
        sc = sc.withColumn("_source_code", lit(""))
    
    if "campaign_name" in sc.columns:
        sc = sc.withColumn("_camp_name",
            coalesce(trim(col("campaign_name").cast(StringType())), lit("")))
    else:
        sc = sc.withColumn("_camp_name", lit(""))
    
    if "campaign_code" in sc.columns:
        sc = sc.withColumn("_camp_code",
            coalesce(trim(col("campaign_code").cast(StringType())), lit("")))
    else:
        sc = sc.withColumn("_camp_code", lit(""))
    
    # --------------------
    # Step 1: client + source_code
    # --------------------
    print("[STEP 1] client + source_code")
    
    lk1 = sc.filter(col("sc_key_curated").isNotNull()).select(
        "_client", "_source_code", "sc_key_curated"
    ).dropDuplicates(["_client", "_source_code"])
    
    out = out.join(
        lk1.withColumnRenamed('sc_key_curated', 'm1'),
        on=["_client", "_source_code"],
        how='left'
    )
    print("   Step 1 join complete")
    
    # --------------------
    # Step 2: client + source_code + campaign_name + campaign_code
    # --------------------
    print("[STEP 2] client + source_code + campaign_name + campaign_code")
    
    lk2 = sc.filter(col("sc_key_curated").isNotNull()).select(
        "_client", "_source_code", "_camp_name", "_camp_code", "sc_key_curated"
    ).dropDuplicates(["_client", "_source_code", "_camp_name", "_camp_code"])
    
    out = out.join(
        lk2.withColumnRenamed('sc_key_curated', 'm2'),
        on=["_client", "_source_code", "_camp_name", "_camp_code"],
        how='left'
    )
    print("   Step 2 join complete")
    
    # --------------------
    # Step 3: client + source_code + campaign_name
    # --------------------
    print("[STEP 3] client + source_code + campaign_name")
    
    lk3 = sc.filter(col("sc_key_curated").isNotNull()).select(
        "_client", "_source_code", "_camp_name", "sc_key_curated"
    ).dropDuplicates(["_client", "_source_code", "_camp_name"])
    
    out = out.join(
        lk3.withColumnRenamed('sc_key_curated', 'm3'),
        on=["_client", "_source_code", "_camp_name"],
        how='left'
    )
    print("   Step 3 join complete")
    
    # --------------------
    # Step 4: FY fallback (UNSOURCED-FY)
    # --------------------
    print("[STEP 4] FY fallback (UNSOURCED-FY)")
    
    sc_uns = sc.filter(col("_camp_code").startswith("UNSOURCED-FY"))
    
    if "gift_fy" in out.columns:
        out = out.withColumn("_gift_fy_norm",
            coalesce(trim(col("gift_fy").cast(StringType())), lit("")))
    else:
        out = out.withColumn("_gift_fy_norm", lit(""))
    
    if "fy" in sc_uns.columns:
        sc_uns = sc_uns.withColumn("_fy_norm",
            coalesce(trim(col("fy").cast(StringType())), lit("")))
    else:
        sc_uns = sc_uns.withColumn("_fy_norm", lit(""))
    
    lk4 = sc_uns.filter(col("sc_key_curated").isNotNull()).select(
        "_client", "_fy_norm", "sc_key_curated"
    ).dropDuplicates(["_client", "_fy_norm"])
    
    # Join on _client and _gift_fy_norm == _fy_norm
    if '_gift_fy_norm' in out.columns:
        out = out.join(
            lk4.withColumnRenamed('sc_key_curated', 'm4').withColumnRenamed('_fy_norm', '_gift_fy_norm'),
            on=['_client', '_gift_fy_norm'],
            how='left'
        )
    print("   Step 4 join complete")
    
    # --------------------
    # Priority combine (Step2 -> Step3 -> Step1 -> Step4)
    # --------------------
    print("[PRIORITY] Applying priority logic (Step2 -> Step3 -> Step1 -> Step4)")
    
    m_cols = [c for c in ['m2', 'm3', 'm1', 'm4'] if c in out.columns]
    if len(m_cols) > 0:
        out = out.withColumn("sc_key_curated", coalesce(*[col(c) for c in m_cols]))
    
    # Cleanup
    cols_to_drop = [c for c in out.columns if c.startswith("_")]
    cols_to_drop.append("sc_key_match_method")
    for mc in m_cols:
        if mc in out.columns:
            cols_to_drop.append(mc)
    for c in cols_to_drop:
        if c in out.columns:
            out = out.drop(c)
    
    print("[COMPLETE] Backfill complete\n")
    
    return out

# COMMAND ----------

# DBTITLE 1,def process_trx_prep_data_pyspark
def process_trx_prep_data_pyspark(df, client, column_renames):
    """
    PySpark version - Original lines 1016-1127
    Prepare transaction data for loading
    """
    # Pandas: df = database_headers(df)
    df = database_headers_pyspark(df)
    df = remove_client_prefix_pyspark(df, client)
    
    # Handle duplicated columns - PySpark doesn't allow duplicate column names
    # Check for duplicates would happen at DataFrame creation
    # Skip this section for PySpark
    
    # Handle various forms of null columns
    # Pandas: df[col] = df[col].replace(r"\bnan\b",np.nan,regex=True)
    # PySpark: Replace 'nan' and 'None' strings with null
    # Single select instead of per-column withColumn loop to reduce Spark plan complexity
    df = df.select([
        when(col(c).cast(StringType()).isin('nan', 'None', ''), lit(None)).otherwise(col(c)).alias(c)
        for c in df.columns
    ])
    
    # Add client
    # Pandas: df['client']=client
    df = df.withColumn('client', lit(client))
    
    # Fix donor_id format
    # Pandas: df["donor_id"] = df["donor_id"].astype(str).str.replace(".0", "", regex=False).replace({"nan": np.nan})
    df = df.withColumn("donor_id",
        regexp_replace(col("donor_id").cast(StringType()), r"\.0$", ""))
    df = df.withColumn("donor_id",
        when(col("donor_id") == "nan", lit(None))
        .otherwise(col("donor_id")))
    
    # Fix dates
    if 'first_gift_date' not in df.columns:
        # Pandas: df['first_gift_date']=pd.NaT
        df = df.withColumn('first_gift_date', lit(None).cast(TimestampNTZType()))
    
    # Cast date columns to timestamp (already cleaned and cast in load_parquet_pyspark)
    # Pandas: df['gift_date'] = pd.to_datetime(df['gift_date'], errors='coerce')
    # Pandas: df['first_gift_date'] = pd.to_datetime(df['first_gift_date'], errors='coerce')
    # No additional casting needed - already TimestampType from load_parquet_pyspark
    
    # Add date processed
    if 'data_processed_at' not in df.columns:
        # Pandas: df['data_processed_at']=np.datetime64("NaT")
        df = df.withColumn('data_processed_at', lit(None).cast(TimestampType()))
    
    # Column renames for consistency
    for k, v in column_renames.items():
        if v in df.columns and k in df.columns:
            # Both source and target exist (e.g. bronze added a native column
            # that overlaps with a rename target).  Drop the source, keep the
            # native target, and cast boolean → "0"/"1" string to match the
            # original Azure output.
            logger.warning(f'Both {k} and {v} present \u2014 dropping {k}, keeping native {v}')
            print(f'Resolve: both {k} and {v} present \u2014 dropping {k}, keeping native {v}')
            df = df.drop(k)
            if dict(df.dtypes).get(v) == 'boolean':
                df = df.withColumn(v,
                    when(col(v) == True, lit('1')).otherwise(lit('0')))
                logger.info(f'Cast {v} boolean \u2192 string 0/1')
        elif k in df.columns:
            print(f'Update: {k} to {v}')
            df = df.withColumnRenamed(k, v)
        else:
            print(f'Skip: {k}')
    
    # Deduplicate by gift id
    if 'gift_id' in df.columns:
        # Pandas: df = df.drop_duplicates('gift_id')
        df = df.dropDuplicates(['gift_id'])
    else:
        # Pandas: df = df.drop_duplicates()
        # Pandas: df['gift_id'] = [i for i in range(df.shape[0])]
        df = df.dropDuplicates()
        df = df.withColumn('gift_id',
            row_number().over(Window.orderBy(monotonically_increasing_id())))
    
    # Add donor id if not in there (placeholder)
    if 'donor_id' not in df.columns:
        logger.warning('Create definition of a donor')
    
    # Remove empty rows
    # Pandas: df = df[(df['donor_id'].notna()) & (df['gift_date'].notna()) & (df['gift_amount'].notna())]
    # NOTE: Bronze ingestion may cast NULL dates to epoch (1970-01-01).
    # The original pandas pipeline dropped NaT via notna(). isNotNull() handles
    # real NULLs; additionally exclude any timestamp on 1970-01-01 (regardless
    # of sub-second precision) to catch bronze NULL-casts while preserving
    # legitimate pre-1971 gifts.
    _EPOCH_DATE = lit('1970-01-01').cast('date')
    df = df.filter(
        col('donor_id').isNotNull() &
        col('gift_date').isNotNull() &
        (col('gift_date').cast('date') != _EPOCH_DATE) &
        col('gift_amount').isNotNull()
    )
    
    # Add empty cols as necessary, cast to the type defined in JSON schema
    # PERF: Replaces 58+ sequential withColumn calls (~3.5s each on serverless)
    all_cols = list(set(gift_dims + donor_dims))
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        logger.info(f'Adding {len(missing_cols)} missing columns in single select')
    
    # Combined schema dict: column_name -> type_string (e.g. 'gift_date' -> 'TimestampNTZType')
    schema_types = {**table_schema['DonorTable'], **table_schema['GiftTable'], **table_schema['SourceCodeTable']}
    
    # Build the select: existing cols + missing cols cast to their JSON-defined type
    select_exprs = [col(c) for c in df.columns]
    for c in missing_cols:
        col_type_str = schema_types.get(c)
        spark_type = SPARK_TYPE_MAP.get(col_type_str) if col_type_str else None
        if spark_type:
            select_exprs.append(lit(None).cast(spark_type).alias(c))
        else:
            select_exprs.append(lit(None).alias(c))
    df = df.select(*select_exprs)
    
    # REMOVED: donor_key (dense_rank) \u2014 replaced by donor_id + client as composite PK
    # (see SILVER_SCHEMA_INCREMENTAL_PLAN.md S7 \u2014 business confirmed no downstream usage)
    
    # REMOVED: gift_key (dense_rank) \u2014 produces wrong values on incremental batches
    # and is not used as the MERGE key (gift_bronze_sk is the merge key)
    
    # Gift Month & CY
    # Pandas: df['gift_month'] = df['gift_date'].dt.month.astype('Int64')
    df = df.withColumn('gift_month', month(col('gift_date')).cast(LongType()))
    # Pandas: df['gift_cy'] = df['gift_date'].dt.year.astype('Int64')
    df = df.withColumn('gift_cy', year(col('gift_date')).cast(LongType()))
    
    # Add sc_key_curated column to gift table
    # Pandas: df['sc_key_curated'] = ''
    df = df.withColumn('sc_key_curated', lit(''))
    
    # Pandas: sc_curated = sql("select * from curated.source_code")
    spark = _get_spark()
    curated_sc_path = get_curated_table_path('source_code')
    sc_curated_spark = spark.table(curated_sc_path)
    
    # Pandas: df = trx_backfill_sc_key_curated(df, sc_curated)
    df = trx_backfill_sc_key_curated_pyspark(df, sc_curated_spark)
    
    logger.info('Data Prepped')
    return df

# COMMAND ----------

# DBTITLE 1,def qa_transactions_pyspark
def _qa_transaction_totals_pyspark(df, client, gift_date_count):
    """PySpark version - Original lines 1135-1144 (count from combined agg).
    MODIFIED: Removed gift_date_count <= 1 gate — on incremental batches
    a small count is expected and should not block the pipeline.
    """
    check_level = 'error'
    detail = f'Record count: {gift_date_count}'
    add_to_error_table(client, 'Transactions.parquet volume', 'Transactions.parquet', detail, 0, check_level)
    return 0, 0

def _qa_keys_pyspark(df, client, missing_donor_id, missing_gift_id):
    """PySpark version - Original lines 1147-1154 (counts from combined agg)."""
    add_to_error_table(client, 'IDs Present', 'Transactions.parquet',
                      f'{missing_donor_id} records missing donor_id',
                      missing_donor_id > 0, 'warning')
    add_to_error_table(client, 'IDs Present', 'Transactions.parquet',
                      f'{missing_gift_id} records missing gift_id',
                      missing_gift_id > 0, 'warning')

def qa_transactions_pyspark(df, client):
    """PySpark version - Original lines 1159-1165"""
    if df.limit(1).count() == 0:
        logger.info('[QA] qa_transactions_pyspark: DataFrame is empty — skipping QA')
        return 0, 0
    _qa_stats = df.agg(
        spark_sum(when(col('gift_date').isNotNull(), lit(1)).otherwise(lit(0))).alias('_gdc'),
        spark_sum(when(col('donor_id').isNull(), lit(1)).otherwise(lit(0))).alias('_md'),
        spark_sum(when(col('gift_id').isNull(), lit(1)).otherwise(lit(0))).alias('_mg'),
    ).first()
    gift_date_count = int(_qa_stats['_gdc'] or 0)
    missing_donor_id = int(_qa_stats['_md'] or 0)
    missing_gift_id = int(_qa_stats['_mg'] or 0)
    _qa_transaction_totals_pyspark(df, client, gift_date_count)
    _qa_keys_pyspark(df, client, missing_donor_id, missing_gift_id)
    warnings, errors = query_qa_errors(client, ['Transactions.parquet'])
    
    return warnings, errors

# COMMAND ----------

# DBTITLE 1,def process_transactions_pyspark
def process_transactions_pyspark(client):
    """
    PySpark version - Original lines 1170-1280
    Main transaction processing orchestrator
    """
    _mode = get_table_mode('gift')
    logger.info(f'Transactions started (mode={_mode})')
    schema = get_schema(client)
    
    # Load & apply suppressions
    df = process_trx_apply_suppressions_pyspark(client)
    _log_phase('trx: after load_parquet + suppressions', client)
    etl2_status_entry(client, 'Transaction Processing: Loaded & Suppressed')
    logger.info('Loaded & Suppressed')
    
    # Column Mappers
    if 'Gift_ID' in df.columns:
        df = df.withColumnRenamed('Gift_ID', 'GiftID')
    
    # Deduplicate by gift id
    if 'GiftID' in df.columns:
        df = df.withColumn('GiftID',
            when((col('GiftID').cast(StringType()) == 'nan') |
                 (col('GiftID').cast(StringType()) == 'None'),
                 lit(None))
            .otherwise(col('GiftID')))
        
        if df.filter(col('GiftID').isNull()).limit(1).count() == 0:
            df2 = df.dropDuplicates(['GiftID'])
        else:
            df2 = df.dropDuplicates()
            df2 = df2.withColumn('row_num',
                row_number().over(Window.orderBy(monotonically_increasing_id())))
            df2 = df2.withColumn('GiftID',
                when(col('GiftID').isNull(),
                     concat(lit('_'), col('row_num').cast(StringType())))
                .otherwise(col('GiftID').cast(StringType())))
            df2 = df2.drop('row_num')
    else:
        df2 = df.dropDuplicates()
        df2 = df2.withColumn('GiftID',
            concat(lit('_'),
                   row_number().over(Window.orderBy(monotonically_increasing_id())).cast(StringType())))
    
    # Free memory
    df = None
    
    etl2_status_entry(client, 'Transaction Processing: Gift IDs Handled')
    logger.info('Gift IDs Handled')
    
    # Exclude $0 gifts but keep negatives
    df2 = df2.filter(coalesce(col('GiftAmount').cast(DoubleType()), lit(0)) != 0)
    
    # Rename cp cols
    if 'CPMapper' in schema.keys():
        for old_name, new_name in schema['CPMapper'].items():
            if old_name in df2.columns:
                df2 = df2.withColumnRenamed(old_name, new_name)
        etl2_status_entry(client, 'Transaction Processing: Rename cp cols based on CPMapper')
        logger.info('Rename cp cols based on CPMapper')
    
    _log_phase('trx: after dedupe + $0 filter + CPMapper', client)
    
    # Add FH columns
    if 'FileHealth' in schema:
        df3 = process_trx_add_fh_filters_pyspark(df2, client)
        # BUG 13: Remove epoch artifact rows (1970-01-01) AFTER FH filters are applied.
        # GiftHistory/DonorGroup already exclude these via conditional aggregation (cell 30).
        # Removing here prevents sc_find_missing from creating spurious FY=1970 SC entries
        # and ensures downstream processing never sees these artifacts.
        df3 = df3.filter(col('GiftDate').cast('date') != lit('1970-01-01').cast('date'))
        df3 = _safe_cache(df3, 'df3_fh')
        etl2_status_entry(client, 'Transaction Processing: FH Filters Added')
        logger.info('FH Filters applied')
    else:
        df3 = df2
        logger.info('FH Filters skipped (no FileHealth in schema)')
    
    df2 = None
    _log_phase('trx: after FileHealth filters', client)
    
    # Force-reload client source_code module to avoid stale workspace FS cache
    import importlib as _il
    for _key in [f'clients.{client.lower()}.source_code', f'clients.{client}.source_code']:
        if _key in sys.modules:
            _il.reload(sys.modules[_key])
    
    # Clean source code column
    # NOTE: df4 is lazy and may still reference df3's temp table (via joins
    # in client update_codes). Do NOT unpersist df3 here — defer until df5
    # is materialized so the temp table stays available for plan resolution.
    df4 = update_client_codes(df3, client)
    etl2_status_entry(client, 'Transaction Processing: Source Codes Cleaned')
    logger.info('Source Codes Processed (update_client_codes)')
    _log_phase('trx: after update_client_codes', client)
    
    # Camp perf actions
    if not apply_client_skip_cp_filters(client):
        df5 = process_trx_add_cp_filters_pyspark(df4, client)
        df5 = _safe_cache(df5, 'df5_cp')
        etl2_status_entry(client, 'Transaction Processing: CP Filters Added')
        logger.info('CP Filters applied')
    else:
        df5 = df4
        logger.info('CP Filters skipped (client override)')
    
    # Now safe to unpersist df3 — df5 is materialized (or df4 is df5 if cp skipped)
    _safe_unpersist(df3)
    df3 = None
    df4 = None
    _log_phase('trx: after CP filters', client)
    
    # Clean and prep
    df6 = process_trx_prep_data_pyspark(df5, client, column_renames)
    _safe_unpersist(df5)
    df5 = None
    etl2_status_entry(client, 'Transaction Processing: Structure Standardized & Prepared for Use')
    logger.info('Structure Standardized & Prepared for Use (process_trx_prep_data)')
    _log_phase('trx: after prep data', client)
    
    # Add QA check
    if not apply_client_skip_trx_qa(client):
        warnings, errors = qa_transactions_pyspark(df6, client)
    else:
        warnings, errors = 0, 0
    _log_phase('trx: after transaction QA', client)
    
    # Store in Delta (silver transactions table)
    if errors == 0:
        df6 = df6.withColumn('gift_id', col('gift_id').cast(StringType()))
        
        table_name = 'transactions'
        df6_db = database_headers_pyspark(df6)
        sql_import_pyspark(df6_db, table_name, overwrite_or_append='overwrite')
        etl2_status_entry(client, 'Transaction Processing: Transactions stored in Delta')
        logger.info('Transactions stored in Delta')
    else:
        etl2_status_entry(client, f'Transaction Processing: FAILED, {errors} errors')
        e = 'Transaction Processing Failed QA. See etl2_qa_results for details'
        notification_exception_raised(client, code_location='process_transactions', error_message=e)
        raise Exception(e)
    
    # Update trx mx data contained date
    _mgd_row = df6.select(spark_max('gift_date').alias('_mgd')).first()
    max_gift_date = _mgd_row['_mgd'] if _mgd_row is not None else None
    update_max_date(client, max_gift_date, 'trx_parquet')
    _log_phase('trx: end', client)
    
    return df6

# COMMAND ----------

# DBTITLE 1,def universal_gift_logic_pyspark
def universal_gift_logic_pyspark(df, client):
    """
    PySpark version - Original lines 1290-1311
    Universal gift table column additions
    """
    # Universal Gift table column add #1
    # Add fuse_tracked_campaign column
    # Load list of unique source codes
    sc = load_sc_pyspark(client)
    
    # Pandas: df["fuse_tracked_campaign"] = df["campaign_code"].isin(sc["CampaignCode"])
    # PySpark: left semi-join avoids collecting distinct codes to the driver
    fuse_codes = sc.select(col('CampaignCode').alias('_fuse_cc')).distinct()
    df = df.join(fuse_codes, col('campaign_code') == col('_fuse_cc'), 'left')
    df = df.withColumn('fuse_tracked_campaign', col('_fuse_cc').isNotNull()).drop('_fuse_cc')
    
    logger.warning("fuse_tracked_campaign column added")
    
    # Universal Gift table column add #2 - Add donor gift number column
    # Pandas: df['donor_gift_number'] = df.sort_values(['donor_id', 'gift_date', 'gift_amount'], ascending=[True, True, False]).groupby('donor_id', sort=False).cumcount() + 1
    # PySpark: Use row_number with window
    window_spec = Window.partitionBy('donor_id').orderBy(
        col('gift_date').asc(),
        col('gift_amount').desc()
    )
    df = df.withColumn('donor_gift_number', row_number().over(window_spec).cast(LongType()))
    
    return df

# COMMAND ----------

# DBTITLE 1,def qa_gift_pyspark
def _qa_gift_comp_to_stg_pyspark(gift, client):
    """PySpark version - Original lines 1421-1489
    REFACTORED (Phase 2): In incremental mode, reads the same incremental
    slice of bronze (using the watermark) so the source-vs-target comparison
    is apples-to-apples. In full mode, preserves original behavior.
    """
    schema = get_schema(client)
    
    # Delete checks for this Client+Dataset combo
    tbl = get_metadata_table_path('etl2_qa_results')
    sql_exec_only(f"DELETE FROM {tbl} WHERE client='{client}' AND dataset='Gift'")
    
    # Load staging data — incremental or full depending on mode
    if is_incremental_enabled() and get_table_mode('gift') == 'incremental':
        # Read the SAME incremental slice that was processed
        last_wm, _ = get_watermark(_get_spark(), client, 'gift', 'dbo_gift')
        stg = load_parquet_pyspark(client, last_watermark=last_wm)
        logger.info(f'[INCREMENTAL QA] Comparing gift batch against bronze slice (watermark={last_wm})')
    else:
        # Full load — read entire bronze
        stg = load_parquet_pyspark(client)
    
    # Filter to GiftAmount > 0 (same filter applied during processing)
    stg = stg.filter(coalesce(col('GiftAmount').cast(DoubleType()), lit(0)) > 0)
    
    # Get date range from staging
    stg_agg = stg.agg(spark_min('GiftDate').alias('mn'), spark_max('GiftDate').alias('mx')).first()
    stg_min = stg_agg['mn']
    stg_max = stg_agg['mx']
    
    mn = stg_min
    mx = stg_max
    
    # Compare within the same date window
    stg_comp = stg.filter((col('GiftDate') >= mn) & (col('GiftDate') <= mx))
    gift_comp = gift.filter((col('gift_date') >= mn) & (col('gift_date') <= mx))
    
    # One cross-join agg for both sums
    _sums = (
        stg_comp.agg(spark_sum('GiftAmount').alias('stg_sum'))
        .crossJoin(gift_comp.agg(spark_sum('gift_amount').alias('gift_sum')))
        .first()
    )
    stg_ga = round(float(_sums['stg_sum'] or 0), 0)
    g_ga = round(float(_sums['gift_sum'] or 0), 0)
    fh_ga = stg_ga
    
    check = 'Gift Amount compared to Staged for FH'
    fail = fh_ga > g_ga
    error_description = f'FH Gift Amount: {fh_ga}, Gift Gift Amount: {g_ga}' if fail else ''
    check_level = 'warning'
    add_to_error_table(client, check, 'Gift', error_description, fail, check_level)
    
    check = 'Gift Amount compared to CP'
    fail = stg_ga > g_ga
    error_description = f'CP Gift Amount: {stg_ga}, Gift Gift Amount: {g_ga}' if fail else ''
    check_level = 'warning'
    add_to_error_table(client, check, 'Gift', error_description, fail, check_level)

def _qa_gift_scs_pyspark(gift, client):
    """PySpark version - Original lines 1494-1498"""
    # Single agg: distinct non-null source_code rows (same as drop_duplicates + notna sum)
    cnt = int(
        gift.select('source_code')
        .dropDuplicates()
        .filter(col('source_code').isNotNull())
        .agg(spark_count(lit(1)).alias('_cnt'))
        .first()['_cnt']
        or 0
    )
    add_to_error_table(
        client,
        'Source Codes Processed',
        'Gift',
        f'{cnt} Source Codes present',
        cnt == 0,
        'warning',
    )
    return 0, (1 if cnt == 0 else 0)

def qa_gift_pyspark(gift, client):
    """PySpark version - Original lines 1503-1511"""
    schema = get_schema(client)
    _qa_gift_comp_to_stg_pyspark(gift, client)
    
    if 'SourceCodeProcessing' in schema['ReportMenu'].keys():
        _qa_gift_scs_pyspark(gift, client)
    
    warnings, errors = query_qa_errors(client, ['Gift'])
    return warnings, errors

# COMMAND ----------

# DBTITLE 1,def process_gift_table_pyspark
def process_gift_table_pyspark(df, client):
    """
    PySpark version - Process and load gift table.
    REFACTORED: Supports incremental MERGE via silver_incremental module.
    - Full load: schema enforce → overwrite (same as before)
    - Incremental: schema enforce → schema gate → MERGE on gift_bronze_sk → deletes
    
    Returns:
        tuple: (gift_df, recompute_donor_ids)
        - recompute_donor_ids: DataFrame of (donor_id, client) whose gifts were
          deleted and therefore need donor-metric recalculation.
          None in full-load mode or when no deletes are pending.
    """
    _mode = get_table_mode('gift')
    logger.info(f'Gifts Started (mode={_mode})')
    
    recompute_donor_ids = None  # Populated if gift deletes occur
    
    # Create dataframe with schema
    gift_schema = create_schema_from_dict(gift_dims, gift_def)
    gift = spark.createDataFrame([], gift_schema)
    
    # Remove columns from dims if they don't exist in this particular client's dataset
    existing_gift_cols = [c for c in gift_dims if c in df.columns]
    tmp = df.select(*existing_gift_cols)
    gift = gift.unionByName(tmp, allowMissingColumns=True)
    
    tmp = None
    
    # Add universal gift columns
    gift = universal_gift_logic_pyspark(gift, client)
    logger.info('universal_gift_logic applied')
    
    # Preserve original bronze data_processed_at; fall back to current timestamp for nulls
    gift = gift.withColumn('data_processed_at',
        coalesce(col('data_processed_at'), lit(ts_now()).cast(TimestampType())))
    
    # PERF FIX: Cache gift after all transforms
    gift = _safe_cache(gift, 'gift')
    _gift_batch_count = gift.count()
    logger.info(f'Gift batch: {_gift_batch_count:,} rows to write ({_mode} mode)')
    
    table_name = 'gift'
    
    apply_client_after_gift_before_qa(gift, client)
    
    # ── Schema enforce (JSON as single source of truth) ──
    gift_json_schema = _load_schema_from_json('gift')
    formatted = database_headers_pyspark(gift)
    if gift_json_schema:
        formatted = enforce_full_json_schema(formatted, gift_json_schema)
    
    import time
    logger.info(f'[{datetime.now()}] Writing gift to DBO')
    start_time = time.time()
    
    if _mode == 'incremental':
        # ── Incremental path ──
        dbo_path = get_dbo_table_path(table_name)
        
        # Ensure table exists from JSON (replaces template)
        if gift_json_schema:
            ensure_silver_table_from_json(spark, dbo_path, gift_json_schema, primary_key='gift_bronze_sk')
        
        # Schema gate + MERGE on gift_bronze_sk
        action = apply_schema_gate(
            spark, formatted, dbo_path, gift_json_schema or {},
            merge_key='gift_bronze_sk', is_composite_key=False
        )
        
        # ── Capture donor_ids that need recalculation due to gift deletes ──
        # Must run BEFORE process_deletes_from_lookup, because after
        # deletion we can no longer look up donor_ids for the removed gifts.
        # These donors are NOT deleted — their metrics are recalculated
        # from the remaining gift history. Only donors with zero remaining
        # gifts are deleted (handled by process_orphan_deletes in donor step).
        last_wm, last_del_ts = get_watermark(spark, client, 'gift', 'dbo_gift')
        table_name_filter = get_bronze_table_name_filter(client, 'gift')
        
        _del_lookup_tbl = _get_metadata_path("delete_lookup")
        if spark.catalog.tableExists(_del_lookup_tbl):
            _del_conds = []
            if table_name_filter:
                _del_conds.append(f"table_name = '{table_name_filter}'")
            if last_del_ts is not None:
                _del_conds.append(f"delete_ts > TIMESTAMP '{last_del_ts}'")
            _del_where = " AND ".join(_del_conds) if _del_conds else "1=1"
            _pending_pks = spark.sql(
                f"SELECT effective_pk FROM {_del_lookup_tbl} WHERE {_del_where}"
            )
            if _pending_pks.count() > 0:
                # Look up donor_ids from gift silver BEFORE deletion removes them
                recompute_donor_ids = (
                    spark.table(dbo_path)
                    .join(_pending_pks,
                          col('gift_bronze_sk') == col('effective_pk'),
                          'left_semi')
                    .select('donor_id', 'client')
                    .distinct()
                )
                recompute_donor_ids = _safe_cache(recompute_donor_ids, 'recompute_donors')
                _recomp_ct = recompute_donor_ids.count()
                logger.info(f'[DELETE] Captured {_recomp_ct} donor_ids for recalculation from pending gift deletes')
        
        # Process deletes from delete_lookup (with delete_ts window)
        new_del_ts = process_deletes_from_lookup(
            spark, dbo_path, 'gift_bronze_sk', last_del_ts,
            table_name_filter=table_name_filter
        )
        
        # Advance DBO gift watermark + log DBO total in one read
        _wm_stats = spark.table(dbo_path).agg(
            spark_max('update_ts').alias('_wm'),
            spark_count(lit(1)).alias('_cnt')
        ).first()
        new_wm = _wm_stats['_wm']
        _dbo_total = int(_wm_stats['_cnt'])
        advance_watermark(spark, client, 'gift', 'dbo_gift',
                         new_watermark=new_wm, new_delete_ts=new_del_ts)
        
        logger.info(f'Gift DBO {action}: batch={_gift_batch_count:,} → DBO total={_dbo_total:,}')
    else:
        # ── Full load path ──
        # Set audit timestamps: update_ts is only set by MERGE in incremental mode,
        # so we must populate it explicitly for full load writes.
        formatted = formatted.withColumn('update_ts', current_timestamp())
        formatted = formatted.withColumn('ingest_ts', current_timestamp())
        
        sql_import_pyspark(formatted, table_name, overwrite_or_append='overwrite')
        
        # ── Seed watermark for future incremental runs ──
        dbo_path = get_dbo_table_path(table_name)
        new_wm = spark.table(dbo_path).agg(spark_max('update_ts')).first()[0]
        advance_watermark(spark, client, 'gift', 'dbo_gift', new_watermark=new_wm)
        logger.info(f'Gift DBO full-load: {_gift_batch_count:,} rows written, watermark seeded → {new_wm}')
    
    formatted = None
    
    elapsed_seconds = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    
    logger.info(f'[{datetime.now()}] Gift DBO Complete (took {minutes}m {seconds}s)')
    etl2_status_entry(client, 'BI Table Processing: Gift Table Loaded')
    
    # QA
    logger.info("Gift QA Run Started")
    if not apply_client_skip_gift_qa(client):
        qa_gift_pyspark(gift, client)
    logger.info("Gift QA Run Complete")
    
    # Clean up cache
    _safe_unpersist(gift)
    
    logger.info('Gifts Processed')
    return gift, recompute_donor_ids

# COMMAND ----------

# DBTITLE 1,def _donor_calc_columns_pyspark
def _donor_calc_columns_pyspark(trx, donor, client):
    """
    PySpark version - Calculate donor-level columns from transaction data.
    REFACTORED: Uses donor_id + client as composite key (replaced donor_key).
    FIX 1: Replaced min(struct) with row_number() for first-gift extraction.
           Tie-breaker: gift_id ASC — verified against full 5,478-donor tied
           population: 100% match with reference data.
    FIX 2: Replaced max(struct) with row_number() for last-gift geography.
           Tie-breaker: gift_id DESC (pandas keep='last' picks largest gift_id).
    
    Source data order is gift_id ASC. Therefore:
      - keep='first' → picks first row  → smallest gift_id → gift_id ASC
      - keep='last'  → picks last row   → largest gift_id  → gift_id DESC
    
    Returns:
        tuple: (donor DataFrame, trx_distinct_donor_count for QA)
    """
    DONOR_KEYS = ['donor_id', 'client']
    DONOR_KEY_SET = set(DONOR_KEYS)
    
    # Helper: join with overlap handling (donor schema pre-creates null columns)
    def _join_with_overlap(left, right, join_keys=None):
        if join_keys is None:
            join_keys = DONOR_KEYS
        jk_set = set(join_keys) if isinstance(join_keys, list) else {join_keys}
        right_cols = [c for c in right.columns if c not in jk_set]
        overlap = [c for c in right_cols if c in left.columns]
        overlap_set = set(overlap)
        if overlap:
            right_renamed = right.toDF(*[
                c + '_new' if c in overlap_set else c for c in right.columns
            ])
            result = left.join(right_renamed, on=join_keys, how='left')
            result = result.select(*[
                coalesce(col(c + '_new'), col(c)).alias(c) if c in overlap_set
                else col(c)
                for c in result.columns if not c.endswith('_new')
            ])
            return result
        else:
            return left.join(right, on=join_keys, how='left')
    
    major_gift_definition = 10000
    fiscal_month_start = get_schema(client)['firstMonthFiscalYear']
    
    # Determine columns from join_cols that exist in trx (for first-gift extraction)
    orig_cols = [x.replace('join_', '').replace('first_', '') for x in join_cols]
    available_cols = [c for c in orig_cols if c in trx.columns and c not in DONOR_KEY_SET]
    
    # Columns for first-gift (excluding gift_date which is handled separately)
    payload_cols = [c for c in available_cols if c != 'gift_date']
    
    # ---- First-gift extraction via row_number ----
    # Pandas: trx.sort_values(['donor_key','gift_date','gift_amount'], ascending=[True,True,False])
    #            .drop_duplicates(subset='donor_key', keep='first')
    # Exhaustive validation (5,478 tied donors): reference data picks smallest gift_id.
    # gift_id ASC: 100% match | gift_id DESC: 0% match | donor_gift_number: ~49%.
    _first_gift_window = Window.partitionBy('donor_id', 'client').orderBy(
        col('gift_date').asc(),
        col('gift_amount').desc(),
        col('gift_id').asc()    # Tie-breaker: smallest gift_id (verified 100% match)
    )
    first_gift = trx.withColumn('_rn', row_number().over(_first_gift_window)) \
        .filter(col('_rn') == 1) \
        .drop('_rn')
    
    # Build first-gift columns DataFrame
    first_gift_select = ['donor_id', 'client', col('gift_date').alias('first_gift_date')]
    for c in payload_cols:
        first_gift_select.append(col(c).alias('join_' + c))
    first_gift_cols = first_gift.select(*first_gift_select)
    
    # Derive cy, fy from first_gift_date
    first_gift_cols = first_gift_cols.withColumn('join_cy', year(col('first_gift_date')).cast(LongType()))
    first_gift_cols = first_gift_cols.withColumn('join_fy',
        fiscal_from_column_pyspark(first_gift_cols, 'first_gift_date', fiscal_month_start))
    
    # ---- Last-gift extraction via row_number ----
    # Pandas: trx.sort_values(['donor_key','gift_date','gift_amount'])
    #            .drop_duplicates('donor_key', keep='last')
    # All ascending + keep='last': picks last row in source order → largest gift_id.
    _last_gift_window = Window.partitionBy('donor_id', 'client').orderBy(
        col('gift_date').desc(),
        col('gift_amount').desc(),
        col('gift_id').desc()   # Tie-breaker: largest gift_id (keep='last' picks end of ASC order)
    )
    last_gift = trx.withColumn('_rn', row_number().over(_last_gift_window)) \
        .filter(col('_rn') == 1) \
        .drop('_rn') \
        .select('donor_id', 'client',
                col('state').alias('state_last_gift'),
                col('zip_code').alias('zip_code_last_gift'))
    
    # ---- Scalar aggregations via groupBy ----
    scalar_agg = trx.groupBy('donor_id', 'client').agg(
        # Lifespan columns
        spark_min('gift_date').alias('_min_gift_date'),
        spark_max('gift_date').alias('_max_gift_date'),
        spark_count('*').alias('gifts'),
        
        # Gift amount stats
        spark_min('gift_amount').alias('min_gift_amount'),
        spark_max('gift_amount').alias('max_gift_amount'),
        F.mean('gift_amount').alias('donor_average_gift'),
        
        # Major gift dates
        spark_min(when(col('gift_amount') >= major_gift_definition, col('gift_date'))).alias('min_major_gift_date'),
        spark_max(when(col('gift_amount') >= major_gift_definition, col('gift_date'))).alias('max_major_gift_date'),
    )
    
    # ---- Combine first-gift + last-gift + scalar aggregations ----
    combined_agg = first_gift_cols \
        .join(last_gift, on=['donor_id', 'client'], how='inner') \
        .join(scalar_agg, on=['donor_id', 'client'], how='inner')
    
    # Cache combined_agg and capture count for QA (avoids separate trx shuffle later)
    combined_agg = _safe_cache(combined_agg, 'combined_agg')
    trx_distinct_donor_count = combined_agg.count()
    
    # ---- Derive columns from aggregations (unchanged logic) ----
    combined_agg = combined_agg \
        .withColumn('lifespan', ((datediff(col('_max_gift_date'), col('_min_gift_date')) + 1) / 365)) \
        .withColumn('core_donor', when(col('lifespan') >= 2, lit('1')).otherwise(lit('0'))) \
        .withColumn('last_gift_date', to_date(col('_max_gift_date'))) \
        .withColumn('min_major_gift_date', to_date(col('min_major_gift_date'))) \
        .withColumn('max_major_gift_date', to_date(col('max_major_gift_date'))) \
        .drop('_min_gift_date', '_max_gift_date')
    
    # ---- Single join for all donor columns (replaces two separate joins) ----
    donor = _join_with_overlap(donor, combined_agg)
    
    # Unpersist combined_agg after join
    _safe_unpersist(combined_agg)
    
    # Fix dtypes (unchanged)
    donor = donor.withColumn('join_fy', col('join_fy').cast(LongType()))
    if 'join_month' in donor.columns:
        donor = donor.withColumn('join_month', col('join_month').cast(LongType()))
    
    # Add major gift definition
    donor = donor.withColumn('major_gift_definition', lit(major_gift_definition))
    
    print("state_last_gift added to donor table")
    print("zip_code_last_gift added to donor table")
    
    donor = apply_client_donor_post_process(donor, trx, client)
    
    # PERF FIX: Cache donor after all joins to prevent re-evaluation.
    donor = _safe_cache(donor, 'donor')
    print("donor cached after calc columns")
    
    return donor, trx_distinct_donor_count

# COMMAND ----------

# DBTITLE 1,def _donor_csl (no changes - wrapper function)
def _donor_csl(client, donor):
    """No changes - Original lines 1789-1801"""
    try:
        return apply_client_donor_csl(donor, client)
    except Exception as e:
        fn_name = "donor_csl"
        add_to_error_table(client,
                          'Donor CSL Failed',
                          'Donor',
                          f'{fn_name} Exception: {repr(e)}',
                          1,
                          'warning')
        logger.error("Donor CSL Failed: %s", traceback.format_exc())
    return donor

# COMMAND ----------

# DBTITLE 1,def process_donor_table_pyspark
def process_donor_table_pyspark(trx, client, recompute_donor_ids=None):
    """
    PySpark version - Process and load donor table.
    REFACTORED: Uses donor_id + client as composite key (replaced donor_key).
    Supports incremental MERGE via silver_incremental module.
    OPTIMIZED: Reuses trx distinct count from _donor_calc_columns_pyspark,
               adds coalesce before write to reduce small files.
    INCREMENTAL FIX: In incremental mode, reads full gift history for impacted
    donors from the gift silver table (post-merge) so that donor aggregations
    (gifts count, avg gift, first/last gift date, etc.) reflect full history.
    
    For ANY operation on the gift silver table (I, U, or D):
    - I/U: impacted donors come from the batch (trx)
    - D: impacted donors come from recompute_donor_ids (captured in gift processing
      BEFORE the gifts were deleted, so we still know which donors were affected)
    - Full gift history is read for ALL impacted donors to recalculate metrics.
    - Donors whose ALL gifts were deleted end up with zero rows in the gift table
      and are cleaned up by process_orphan_deletes (the ONLY place donor deletion happens).
    
    Args:
        trx: Gift batch DataFrame (I/U records from bronze)
        client: Client identifier
        recompute_donor_ids: DataFrame of (donor_id, client) whose gifts were deleted
                             and therefore need donor-metric recalculation.
                             These donors are NOT deleted — only recalculated.
                             Passed from process_gift_table_pyspark via orchestrator.
                             None when no gift deletes occurred or in full-load mode.
    """
    _mode = get_table_mode('donor')
    logger.info(f'Donors Started (mode={_mode})')
    
    DONOR_MERGE_KEYS = ['donor_id', 'client']
    
    # ── Incremental: build impacted donor set from I + U + D operations ──
    if is_incremental_enabled() and _mode == 'incremental':
        # Donors from batch (Insert / Update operations)
        impacted_donors = trx.select(*DONOR_MERGE_KEYS).distinct()
        
        # Donors whose gifts were deleted (Delete operations) — need recalculation
        if recompute_donor_ids is not None:
            impacted_donors = impacted_donors.unionByName(
                recompute_donor_ids.select(*DONOR_MERGE_KEYS)
            ).distinct()
            logger.info('[INCREMENTAL] Donor: included gift-delete-affected donor_ids for recalculation')
        
        # Read full gift history for ALL impacted donors from gift silver.
        # The gift silver table is already up-to-date: MERGEs and deletes
        # have been applied by process_gift_table_pyspark.
        gift_dbo_path = get_dbo_table_path('gift')
        if spark.catalog.tableExists(gift_dbo_path):
            trx = spark.table(gift_dbo_path).join(
                F.broadcast(impacted_donors),
                on=DONOR_MERGE_KEYS,
                how='left_semi'
            )
            # Cache trx to prevent re-scanning gift DBO on every downstream operation
            trx = _safe_cache(trx, 'donor_trx')
            logger.info(f'[INCREMENTAL] Donor: reading full gift history for impacted donors from {gift_dbo_path}')
        else:
            logger.warning(f'[INCREMENTAL] Gift DBO table {gift_dbo_path} not found — using batch only')
    
    ########## Create Base Donor Table (deduplicated from trx)
    
    # Create dataframe
    donor_schema = create_schema_from_dict(donor_dims, donor_def)
    donor = spark.createDataFrame([], donor_schema)
    
    # Tmp donors
    # Exclude per-gift metadata timestamps that vary across gift rows.
    # In incremental mode, trx contains the full gift history — each donor has
    # multiple gifts with different data_processed_at/ingest_ts/update_ts.
    # dropDuplicates() fails to collapse rows when these differ.
    # These columns are overwritten anyway:
    #   - data_processed_at → replaced by lit(ts_now()) below
    #   - ingest_ts/update_ts → set by the MERGE operation (incremental) or explicitly (full load)
    _ts_cols_exclude = {'data_processed_at', 'ingest_ts', 'update_ts'}
    existing_donor_cols = [c for c in donor_dims if c in trx.columns and c not in _ts_cols_exclude]
    tmp_donor = trx.select(*existing_donor_cols)
    
    # CSL
    tmp_donor = _donor_csl(client, tmp_donor)
    
    # Insert deduplicated copy
    donor = donor.unionByName(tmp_donor.dropDuplicates(), allowMissingColumns=True)
    
    tmp_donor = None
    
    # Add Calc Cols (returns tuple with trx distinct count for QA)
    donor, trx_donor_count = _donor_calc_columns_pyspark(trx, donor, client)
    
    # Unpersist donor_trx cache (no longer needed after calc columns)
    _safe_unpersist(trx)
    
    # Add ts
    donor = donor.withColumn('data_processed_at', lit(ts_now()).cast(TimestampType()))
    
    # QA: Check for missing/duplicate donor_ids
    _dstat = donor.agg(
        spark_sum(when(col('donor_id').isNull(), lit(1)).otherwise(lit(0))).alias('_miss'),
        spark_count(lit(1)).alias('_tot'),
    ).first()
    missing_ct = int(_dstat['_miss'] or 0)
    total_donors = int(_dstat['_tot'] or 0)
    distinct_donors = donor.select('donor_id', 'client').distinct().count()
    missing = missing_ct > 0
    add_to_error_table(
        client,
        'Missing donor_ids',
        'Donor',
        f"{missing_ct} records with missing donors",
        missing,
        'warning'
    )
    dupes = total_donors != distinct_donors
    
    add_to_error_table(
        client,
        'Duplicated donor_ids',
        'Donor',
        f"{total_donors - distinct_donors} donor_ids with conflicting/mismatched donor info",
        dupes,
        'error'
    )
    
    logger.info(f'Donor batch: {total_donors:,} donors to write ({_mode} mode)')
    
    table_name = 'donor'
    etl2_status_entry(client, 'BI Table Processing: Donors Processed')
    
    # ── Schema enforce (JSON as single source of truth) ──
    donor_json_schema = _load_schema_from_json('donor')
    formatted = database_headers_pyspark(donor)
    if donor_json_schema:
        formatted = enforce_full_json_schema(formatted, donor_json_schema)
    
    # PERF: Coalesce before write to reduce small files (798K rows → ~8 files)
    formatted = formatted.coalesce(8)
    
    import time
    logger.info(f'[{datetime.now()}] Writing donor to DBO')
    start_time = time.time()
    
    if _mode == 'incremental':
        # ── Incremental path ──
        dbo_path = get_dbo_table_path(table_name)
        
        # Ensure table exists from JSON (replaces template)
        if donor_json_schema:
            ensure_silver_table_from_json(spark, dbo_path, donor_json_schema)
        
        # Schema gate + MERGE on donor_id + client
        action = apply_schema_gate(
            spark, formatted, dbo_path, donor_json_schema or {},
            merge_key=DONOR_MERGE_KEYS, is_composite_key=True
        )
        
        # Orphan delete: the ONLY place donor rows are hard-deleted.
        # Anti-joins donor table vs gift table — if a donor_id has zero
        # remaining gifts (e.g. ALL gifts were deleted), remove that donor.
        gift_dbo_path = get_dbo_table_path('gift')
        _orphan_del_count = 0
        if spark.catalog.tableExists(gift_dbo_path):
            _orphan_del_count = process_orphan_deletes(
                spark, dbo_path, gift_dbo_path,
                join_keys=DONOR_MERGE_KEYS, target_entity_name='donor'
            )
        _orphan_del_ts = datetime.now() if _orphan_del_count and _orphan_del_count > 0 else None
        
        # Advance DBO donor watermark + log DBO total in one read
        _wm_stats = spark.table(dbo_path).agg(
            spark_max('update_ts').alias('_wm'),
            spark_count(lit(1)).alias('_cnt')
        ).first()
        new_wm = _wm_stats['_wm']
        _dbo_total = int(_wm_stats['_cnt'])
        advance_watermark(spark, client, 'donor', 'dbo_donor', new_watermark=new_wm, new_delete_ts=_orphan_del_ts)
        
        logger.info(f'Donor DBO {action}: batch={total_donors:,} → DBO total={_dbo_total:,}')
    else:
        # ── Full load path ──
        # Set audit timestamps: update_ts is only set by MERGE in incremental mode,
        # so we must populate it explicitly for full load writes.
        formatted = formatted.withColumn('update_ts', current_timestamp())
        formatted = formatted.withColumn('ingest_ts', current_timestamp())
        
        sql_import_pyspark(formatted, table_name, overwrite_or_append='overwrite')
        
        # ── Seed watermark for future incremental runs ──
        dbo_path = get_dbo_table_path(table_name)
        new_wm = spark.table(dbo_path).agg(spark_max('update_ts')).first()[0]
        advance_watermark(spark, client, 'donor', 'dbo_donor', new_watermark=new_wm)
        logger.info(f'Donor DBO full-load: {total_donors:,} donors written, watermark seeded → {new_wm}')
    
    formatted = None
    
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    
    logger.info(f'[{datetime.now()}] Donor DBO Complete (took {minutes}m {seconds}s)')
    etl2_status_entry(client, 'BI Table Processing: Donor Table Loaded')
    
    # QA: Compare donor_id counts between trx and donor
    logger.info("Donor QA Run Started")
    donor_donor_count = donor.select('donor_id', 'client').distinct().count()
    donors_mismatch = trx_donor_count != donor_donor_count
    
    add_to_error_table(
        client,
        'Donor_id counts match Gift',
        'Donor',
        f'Trx: {trx_donor_count} vs Donor: {donor_donor_count}',
        donors_mismatch,
        'error'
    )
    logger.info("Donor QA Run Complete")
    
    logger.info('Donors Processed')
    return donor

# COMMAND ----------

# DBTITLE 1,def process_bi_tables_pyspark
def process_bi_tables_pyspark(dfx, budget_df, client):
    """
    PySpark version - Original lines 1986-2011
    Process all BI tables.
    Passes recompute_donor_ids from gift → donor so that donor recalculation
    covers all three operations (I, U, D) on the gift silver table.
    """
    print('process_bi_tables running')
    print('process_gift_table started')
    _, recompute_donor_ids = process_gift_table_pyspark(dfx, client)
    _log_phase('bi: after process_gift_table', client)
    print('process_gift_table complete')
    
    process_donor_table_pyspark(dfx, client, recompute_donor_ids=recompute_donor_ids)
    _log_phase('bi: after process_donor_table', client)
    # Clean up cached recompute_donor_ids (no longer needed)
    _safe_unpersist(recompute_donor_ids)
    print('process_donor_table complete')
    
    process_budget_table_pyspark(budget_df, client)
    _log_phase('bi: after process_budget_table', client)
    print('process_budget_table complete')
    
    print('query_qa_errors started')
    warnings, errors = query_qa_errors(client, ['Gift', 'Donor', 'Source Code'])
    print('query_qa_errors complete')
    
    if errors == 0:
        etl2_status_entry(client, f'Gift, Donor & SC Processing: Complete, {errors} errors')
        logger.info('Publishing to curated')
        publish_bi_tables(client)
        logger.info('Published to curated')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Master tables: skipped (implement later)')
    else:
        e = 'Transactions did not pass QA'
        etl2_status_entry(client, f'Gift, Donor & SC Processing: FAILED, {errors} errors')
        notification_exception_raised(client, code_location='process_bi_tables', error_message=e)
        raise Exception(e)

# COMMAND ----------

# DBTITLE 1,def process_response_curve_pyspark
def process_response_curve_pyspark(client):
    """
    Straight dump from bronze to silver (DBO + curated).
    Full load with overwriteSchema=true. No schema enforcement.
    One-to-one mapping — all columns flow through as-is.
    Adds ingest_ts and update_ts audit columns.
    Skips gracefully if response_curve_bronze does not exist for the client.
    """
    logger.info('Response Curve Started')

    # Check if bronze table exists before attempting to read
    cfg = _load_etl_config()
    catalog = cfg.get('bronze_catalog', cfg.get('metadata_catalog', 'dev_catalog'))
    table_name = f"{catalog}.{client.lower()}.response_curve_bronze"

    if not spark.catalog.tableExists(table_name):
        logger.info(f'Response Curve: bronze table {table_name} not found for client {client}, skipping')
        return

    # Read full from bronze
    df = _read_from_bronze_pyspark(client, 'response_curve')
    if df is None:
        logger.warning('Response Curve: bronze table read failed, skipping')
        return

    # Add audit timestamps
    df = df.withColumn('ingest_ts', current_timestamp()) \
           .withColumn('update_ts', current_timestamp())

    row_count = df.count()
    logger.info(f'Response Curve: {row_count:,} rows read from bronze')

    # Write to DBO silver (full overwrite)
    dbo_path = get_dbo_table_path('response_curve')
    df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').saveAsTable(dbo_path)
    print(f'Written {row_count:,} rows to {dbo_path}')
    etl2_status_entry(client, 'BI Table Processing: Response Curve DBO Complete')

    # Publish DBO → curated (full overwrite)
    publish_to_curated('response_curve')
    curated_path = get_curated_table_path('response_curve')
    print(f'Published response_curve: {dbo_path} → {curated_path}')
    etl2_status_entry(client, 'BI Table Processing: Response Curve Curated Complete')

    logger.info('Response Curve Complete')

# COMMAND ----------

def update_max_gift_date(client):
    mgd_tbl = get_metadata_table_path('max_gift_dates')
    gift_tbl = get_curated_table_path('gift')
    try:
        sql_exec_only(f"""
          MERGE INTO {mgd_tbl} AS m
          USING (SELECT client, max(gift_date) AS mx FROM {gift_tbl} GROUP BY client) AS g
          ON m.client = g.client
          WHEN MATCHED THEN UPDATE SET m.curated = g.mx
          WHEN NOT MATCHED THEN INSERT (client, curated) VALUES (g.client, g.mx)
        """)
    except Exception:
        sql_exec_only(f"""
          INSERT INTO {mgd_tbl} (client, curated)
          SELECT client, max(gift_date) FROM {gift_tbl} GROUP BY client
        """)


def update_max_date(client, date, col_name):
    tbl = get_metadata_table_path('max_gift_dates')
    cl = str(client).strip() if client else ''
    if date is None:
        logger.warning(f'update_max_date: date is None for client={cl}, col={col_name} \u2014 skipping')
        return
    try:
        sql_exec_only(f"UPDATE {tbl} SET {col_name} = '{date}' WHERE client = '{cl}'")
    except Exception:
        sql_exec_only(f"INSERT INTO {tbl} (client, {col_name}) VALUES ('{cl}', '{date}')")


def _sync_curated_schema(dbo_path, curated_path):
    """Drop columns from curated that were removed from DBO.
    Falls back to full overwrite if ALTER TABLE DROP COLUMN is not supported.
    """
    dbo_cols = set(spark.table(dbo_path).columns)
    cur_cols = set(spark.table(curated_path).columns)
    extra_cols = cur_cols - dbo_cols
    if not extra_cols:
        return

    try:
        _ensure_column_mapping(curated_path)
        for c in extra_cols:
            spark.sql(f"ALTER TABLE {curated_path} DROP COLUMN `{c}`")
        logger.info(f'[CURATED SCHEMA SYNC] Dropped {extra_cols} from {curated_path}')
    except Exception:
        logger.info(f'[CURATED SCHEMA SYNC] DROP COLUMN failed, falling back to full overwrite for {curated_path}')
        df = spark.table(dbo_path)
        df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').saveAsTable(curated_path)
        logger.info(f'[CURATED SCHEMA SYNC] Full overwrite complete for {curated_path}')


def _ensure_column_mapping_all_tables():
    """Set delta.columnMapping.mode='name' on all existing silver tables (DBO + curated).
    No-op per table if already set. Safe to call every run.
    """
    for tbl in ['gift', 'donor', 'source_code', 'budget', 'response_curve']:
        for path_fn in [get_dbo_table_path, get_curated_table_path]:
            path = path_fn(tbl)
            try:
                if spark.catalog.tableExists(path):
                    _ensure_column_mapping(path)
            except Exception as e:
                logger.warning(f'[COLUMN MAPPING] Could not set on {path}: {e}')


def publish_bi_tables(client):
    """Publish DBO tables to curated layer.
    Incremental MERGE for gift/donor, full overwrite for budget/source_code.
    """
    gift_table = 'gift'
    donor_table = 'donor'
    budget_table = 'budget'
    sc_table = 'source_code'

    _ensure_column_mapping_all_tables()

    if get_table_mode('gift') == 'incremental':
        last_wm, _ = get_watermark(spark, client, 'gift', 'curated_gift')
        new_wm = publish_to_curated_incremental(
            gift_table, merge_key='gift_bronze_sk',
            last_curated_watermark=last_wm, is_composite_key=False
        )
        _cur_del_count = publish_curated_delete_sync(gift_table, merge_key='gift_bronze_sk', is_composite_key=False)
        _cur_del_ts = datetime.now() if _cur_del_count and _cur_del_count > 0 else None
        _sync_curated_schema(get_dbo_table_path(gift_table), get_curated_table_path(gift_table))
        advance_watermark(spark, client, 'gift', 'curated_gift', new_watermark=new_wm, new_delete_ts=_cur_del_ts)
    else:
        publish_to_curated(gift_table)
        curated_path = get_curated_table_path(gift_table)
        new_wm = spark.table(curated_path).agg(spark_max('update_ts')).first()[0]
        if new_wm is None:
            new_wm = datetime.now()
        advance_watermark(spark, client, 'gift', 'curated_gift', new_watermark=new_wm)
        logger.info(f'Gift curated full-load: watermark seeded \u2192 {new_wm}')
    etl2_status_entry(client, 'BI Table Processing: Gift published to curated')
    update_max_gift_date(client)

    if get_table_mode('donor') == 'incremental':
        last_wm, _ = get_watermark(spark, client, 'donor', 'curated_donor')
        new_wm = publish_to_curated_incremental(
            donor_table, merge_key=['donor_id', 'client'],
            last_curated_watermark=last_wm, is_composite_key=True
        )
        _cur_del_count = publish_curated_delete_sync(donor_table, merge_key=['donor_id', 'client'], is_composite_key=True)
        _cur_del_ts = datetime.now() if _cur_del_count and _cur_del_count > 0 else None
        _sync_curated_schema(get_dbo_table_path(donor_table), get_curated_table_path(donor_table))
        advance_watermark(spark, client, 'donor', 'curated_donor', new_watermark=new_wm, new_delete_ts=_cur_del_ts)
    else:
        publish_to_curated(donor_table)
        curated_path = get_curated_table_path(donor_table)
        new_wm = spark.table(curated_path).agg(spark_max('update_ts')).first()[0]
        if new_wm is None:
            new_wm = datetime.now()
        advance_watermark(spark, client, 'donor', 'curated_donor', new_watermark=new_wm)
        logger.info(f'Donor curated full-load: watermark seeded \u2192 {new_wm}')
    etl2_status_entry(client, 'BI Table Processing: Donor published to curated')

    publish_to_curated(budget_table)
    etl2_status_entry(client, 'BI Table Processing: Budget published to curated')

    publish_to_curated(sc_table)
    etl2_status_entry(client, 'BI Table Processing: Source code published to curated')

    apply_client_after_publish_bi_tables(client)

# COMMAND ----------

# DBTITLE 1,Config: Select Full Load or Incremental
# ── Load ETL config (single source of truth) ──
# The config file controls whether incremental mode is enabled or not
# via silver_incremental.enabled = true/false.
# No hardcoded flags — just read the config.

CONFIG_PATH = "../config/etl_config_incremental.json"
_cfg = load_etl_config(CONFIG_PATH)

_incremental_enabled = _cfg.get('silver_incremental', {}).get('enabled', False)

if _incremental_enabled:
    print(f"✅ INCREMENTAL MODE: Using {CONFIG_PATH}")
    print(f"   Silver writes → {_cfg.get('silver_catalog', 'unknown')}")
else:
    print(f"📦 FULL LOAD MODE: Using {CONFIG_PATH}")
    print(f"   Silver writes → {_cfg.get('silver_catalog', 'unknown')}")

print(f"   Client(s): {_cfg.get('clients')}")
print(f"   silver_catalog: {_cfg.get('silver_catalog')}")
print(f"   incremental enabled: {_incremental_enabled}")

# Phase timing only needed for incremental debugging
PHASE_TIMING = _incremental_enabled
print(f"   Phase timing: {'ON' if PHASE_TIMING else 'OFF'}")

# COMMAND ----------

# DBTITLE 1,Setup & Logging
import warnings
import os

# Suppress known-noisy library warnings (pandas, pyarrow, etc.)
# but ALLOW RuntimeWarning from ETL client modules so data-affecting
# fallbacks are never silent.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("default", category=RuntimeWarning)

# Clients from config
clients = get_clients_from_config()
if not clients:
    try:
        clients = ast.literal_eval(dbutils.widgets.get('client'))
    except Exception:
        clients = ['FS']

if isinstance(clients, str):
    clients = [clients]

client = clients[0]

# Set up logging — logs go to Shared/logs/ folder (relative to notebook)
try:
    _notebook_dir = os.path.dirname(os.path.abspath(''))
    _log_dir = os.path.join('/Workspace/Users',
        spark.sql('SELECT current_user()').first()[0],
        'Base_table_Generation_code/Shared/logs')
    os.makedirs(_log_dir, exist_ok=True)
except Exception:
    _log_dir = '/tmp'

# Use a NAMED logger to avoid capturing Databricks internal framework messages
logger = logging.getLogger('etl')

# Format: timestamp - level - [function_name] message
_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Console handler (shared across all clients)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)


def _rotate_log_for_client(client_name):
    """Swap the file handler so each client gets its own log file.
    Called at the start of main_pyspark() for each client.
    """
    # Remove all existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    # New per-client file handler
    log_file = f'{_log_dir}/{client_name}_base_table_processing.log'
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(_log_formatter)

    logger.addHandler(fh)
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

    print(f'Log file: {log_file}')


# Initialize logger for first client
_rotate_log_for_client(client)
etl2_status_entry(client, 'Table generation started')

# COMMAND ----------

# DBTITLE 1,Load Table Schemas
table_schema = get_table_schemas()

# Column Renames
column_renames = table_schema['ColumnRenames']

# Bronze to Silver Column Cleaning
bronze_to_silver_renames = table_schema.get('BronzeToSilverColumnRenames', {})

# Gift Table Schemas
gift_def = table_schema['GiftTable']
gift_dims = list(gift_def.keys())

# Source Code
sc_def = table_schema['SourceCodeTable']
sc_dims = list(sc_def.keys())

# Donor Table Schemas
donor_def = table_schema['DonorTable']
donor_dims = list(donor_def.keys())

# Budget Table Schemas
budget_def = table_schema['BudgetTable']
budget_dims = list(budget_def.keys())

# Join Cols
join_cols = table_schema['JoinColumns']
etl2_status_entry(client, 'Table schemas defined')

# COMMAND ----------

# DBTITLE 1,Required Cols for this Client (etl2_column_qa)
def _populate_etl2_column_qa(client, schema):
    """Record required FileHealth column names for the client in etl2_column_qa.
    Ported from original Base Table Generation.py (lines 2428-2443).
    Stores raw schema keys (no normalization — table is informational only).
    """
    if 'FileHealth' not in schema:
        logger.info('etl2_column_qa: No FileHealth in schema, skipping')
        return

    exclude = {'EndofRptPeriod', 'calcDate'}
    cols = [k for k in schema['FileHealth'].keys() if k not in exclude]

    spark = _get_spark()
    data = [(name, client) for name in cols]
    col_qa_schema = StructType([
        StructField('column_name', StringType(), True),
        StructField('client', StringType(), True)
    ])
    df = spark.createDataFrame(data, col_qa_schema)

    tbl = get_metadata_table_path('etl2_column_qa')
    sql_exec_only(f"DELETE FROM {tbl} WHERE client = '{client}'")
    sql_import_pyspark(df, 'etl2_column_qa', 'append')
    logger.info(f'etl2_column_qa: {len(cols)} column names recorded for {client}')


# Run at module level (same as original — after schema loading, before def main)
_client_schema = get_schema(client)
_populate_etl2_column_qa(client, _client_schema)

# COMMAND ----------

# DBTITLE 1,Main Processing Function
def main_pyspark(client):
    try:
        # Set the active client for silver table path resolution
        set_current_client(client)

        # Rotate log file for this client (each client gets its own log)
        _rotate_log_for_client(client)

        logger.info('Start Main')
        _phase_reset()
        _log_phase('main: start', client)

        # Clear stale status entries for this client (ported from ETL2.py)
        etl2_status_clear(client)

        global schema
        schema = get_schema(client)
        
        # Drop orphaned temp tables from any previous interrupted runs
        _cleanup_stale_temp_tables()
        
        ensure_watermark_table(spark)
        logger.info('Silver watermark table initialized')
        
        schema_changed = schema_preflight_check(spark, ['gift', 'donor'])
        if schema_changed:
            logger.warning('SCHEMA PRE-CHECK: Full load override ACTIVATED for this run')
        _log_phase('main: after schema preflight check', client)
        
        dfx = process_transactions_pyspark(client)
        _log_phase('main: after process_transactions', client)
        logger.info('process_transactions Complete')
        
        dfx = _safe_cache(dfx, 'dfx')
        _dfx_count = dfx.count()
        logger.info(f'Bronze \u2192 transactions: {_dfx_count:,} rows read ({get_table_mode("gift")} mode)')
        _log_phase('main: after dfx cache', client)
        
        budget_df = budget_load_and_process_csv_pyspark(client)
        _log_phase('main: after budget_load_and_process_csv', client)
        logger.info('budget_load_and_process_csv Complete')
        
        process_bi_tables_pyspark(dfx, budget_df, client)
        _log_phase('main: after process_bi_tables', client)
        logger.info('process_bi_tables Complete')
        
        _safe_unpersist(dfx)
        
        # Response curve: independent full load, no dependency on gift/donor QA
        process_response_curve_pyspark(client)
        _log_phase('main: after process_response_curve', client)
        
        logger.info('Process Complete')
    
    except ValueError as ve:
        logger.error("ValueError occurred: %s", ve)
        notification_exception_raised(client, code_location='main', error_message=repr(ve))
    
    except Exception as e:
        logger.error("main: %s", traceback.format_exc())
        notification_exception_raised(client, code_location='main', error_message=repr(e))
    
    finally:
        set_force_full_load(False)
        _cleanup_temp_tables()

# COMMAND ----------

# DBTITLE 1,Execute Main
for c in clients:
    main_pyspark(c)