"""CHOA source code validation logic - PySpark only for Databricks."""


def initiate_codes(df, client):
    """PySpark only for Databricks."""
    from pyspark.sql.functions import col, when, concat, substring, lit

    # _SourceCode: if CampaignCode ends with 'A', insert '___' at position 11; else append '___'
    df = df.withColumn('_SourceCode',
        when(substring(col('CampaignCode'), -1, 1) == 'A',
             concat(substring(col('SourceCode'), 1, 11), lit('___'), substring(col('SourceCode'), 12, 100)))
        .otherwise(concat(col('SourceCode'), lit('___'))))
    return df


def _ljust(c, width, fill):
    """PySpark equivalent of pandas str.ljust: pad if shorter, leave unchanged if longer.
    Unlike rpad() which truncates strings exceeding `width`."""
    from pyspark.sql.functions import when, rpad, length
    return when(length(c) < width, rpad(c, width, fill)).otherwise(c)


def update_codes(df, client):
    """PySpark only for Databricks."""
    from pyspark.sql.functions import (
        col, when, concat, substring, lit, upper, rpad, length,
        regexp_replace, sum as spark_sum
    )

    if 'GiftAmount' not in df.columns:
        # Source code table path: aggregate quantity, create synthetic source codes

        # Rename Quantity -> _Quantity
        df = df.withColumnRenamed('Quantity', '_Quantity')

        # Group by SourceCode, sum _Quantity -> Quantity
        qty_agg = df.groupBy('SourceCode').agg(spark_sum('_Quantity').alias('Quantity'))
        df = df.join(qty_agg, on='SourceCode', how='left')

        # Deduplicate on SourceCode
        df = df.dropDuplicates(['SourceCode'])

        # Filter out rows where CampaignCode or PackageCode is null/blank
        df = df.filter(
            col('CampaignCode').isNotNull() & (col('CampaignCode') != '') & (col('CampaignCode') != ' ') &
            col('PackageCode').isNotNull() & (col('PackageCode') != '') & (col('PackageCode') != ' ')
        )

        # Create XXX RFM codes: deduplicate on (CampaignCode, PackageCode)
        xxx_df = df.dropDuplicates(['CampaignCode', 'PackageCode'])
        xxx_df = xxx_df.withColumn('RFMCode', lit('XXX'))
        xxx_df = xxx_df.withColumn('SourceCode',
            concat(col('CampaignCode'), col('PackageCode'), col('RFMCode')))
        cols = ['SourceCode', 'CampaignCode', 'PackageCode', 'RFMCode',
                'CampaignName', 'PackageName', 'PackageCPP', 'MailDate']
        common_cols = [c for c in cols if c in xxx_df.columns]
        df = df.unionByName(xxx_df.select(*common_cols), allowMissingColumns=True)

        # Landing page records: unique CampaignCodes
        land_cols = ['CampaignCode', 'CampaignName', 'MailDate']
        land_df = df.select(*[c for c in land_cols if c in df.columns]).dropDuplicates(['CampaignCode'])
        land_df = land_df.withColumn('PackageCode', lit('LAND'))
        land_df = land_df.withColumn('RFMCode', lit('XXX'))
        land_df = land_df.withColumn('SourceCode',
            concat(col('CampaignCode'), col('PackageCode'), col('RFMCode')))
        df = df.unionByName(land_df, allowMissingColumns=True)

    if 'GiftAmount' in df.columns:
        # Gift/transaction path
        #
        # Use the CRM-provided SourceCode directly. Bronze stores the full PackageID,
        # but CRM SourceCode already reflects the legacy 4-character PackageID truncation
        # used by the old pipeline outputs. Reconstructing from components creates
        # non-legacy codes such as DM0109APSNTSRXXX instead of DM0109APSNTXXX.

        # Fix CRM nan artifacts before padding/uppercasing.
        df = df.withColumn('SourceCode',
            regexp_replace(col('SourceCode'), 'nan', ''))

        # Pad SourceCode to 14 with 'X' (don't truncate longer codes)
        df = df.withColumn('SourceCode', _ljust(col('SourceCode'), 14, 'X'))

        # Rename DM_Acquisition_List -> ListCode for downstream compatibility
        if 'DM_Acquisition_List' in df.columns:
            df = df.withColumnRenamed('DM_Acquisition_List', 'ListCode')

    # _SourceCode: if CampaignCode ends with 'A', insert '___' at position 11; else append '___'
    df = df.withColumn('_SourceCode',
        when(substring(col('CampaignCode'), -1, 1) == 'A',
             concat(substring(col('SourceCode'), 1, 11), lit('___'), substring(col('SourceCode'), 12, 100)))
        .otherwise(concat(col('SourceCode'), lit('___'))))

    # Clean trailing .0 from ListCode and SourceCode
    for c in ['ListCode', 'SourceCode']:
        if c in df.columns:
            df = df.withColumn(c, regexp_replace(col(c).cast('string'), r'\.0+$', ''))

    # Uppercase SourceCode for SC table path ONLY.
    # Old ETL: general update_codes() uppercased SC; CHOA_update_codes() was a no-op
    # for gifts — so gift source codes preserved original case from CRM/parser.
    if 'GiftAmount' not in df.columns:
        if 'SourceCode' in df.columns:
            df = df.withColumn('SourceCode', upper(col('SourceCode')))

    # Set PackageID/PackageCode to 'LAND' where SourceCode[7:11] == 'LAND'
    # PySpark: substring(col, 8, 4) (1-indexed) = pandas str[7:11] (0-indexed)
    if 'GiftAmount' in df.columns:
        df = df.withColumn('PackageID',
            when(substring(col('SourceCode'), 8, 4) == 'LAND', lit('LAND'))
            .otherwise(col('PackageID')))
    else:
        df = df.withColumn('PackageCode',
            when(substring(col('SourceCode'), 8, 4) == 'LAND', lit('LAND'))
            .otherwise(col('PackageCode')))

    return df


def add_filters(df, client):
    """PySpark only for Databricks."""
    from pyspark.sql.functions import col, when, lit

    # ReportChannel based on CampaignCode prefix + suffix patterns
    # Old ETL: _CHOA_ReportChannelFilter → creates ReportChannel on SC table
    df = df.withColumn('ReportChannel',
        when((col('CampaignCode').startswith('DM')) &
             (col('CampaignCode').endswith('R') | col('CampaignCode').endswith('E')),
             lit('Renewal'))
        .when((col('CampaignCode').startswith('DM')) &
              col('CampaignCode').endswith('L'),
              lit('Lapsed'))
        .when((col('CampaignCode').startswith('DM')) &
              col('CampaignCode').endswith('A'),
              lit('ACQ'))
        .when((col('CampaignCode').startswith('SO')) &
              (col('CampaignCode').endswith('S') | col('CampaignCode').endswith('R') | col('CampaignCode').endswith('W')),
              lit('Leadership'))
        .when((col('CampaignCode').startswith('CR')) &
              (col('CampaignCode').endswith('T') | col('CampaignCode').endswith('F')),
              lit('CAT/Aflac'))
        .when(col('CampaignCode').startswith('CRD'),
              lit('CAT/Aflac'))
    )
    return df


def dataset_filters(df, client):
    """PySpark only for Databricks. Inlines GiftFiscal (CHOA firstMonthFiscalYear=1)."""
    from pyspark.sql.functions import col, year

    # CHOA firstMonthFiscalYear = 1 -> fiscal year = calendar year
    df = df.withColumn('GiftFiscal', year(col('GiftDate')).cast('int'))
    return df
