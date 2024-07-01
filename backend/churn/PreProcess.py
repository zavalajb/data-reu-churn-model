# Class for preproccessing the data

import logging
# SQL libraries
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, BooleanType, ArrayType, LongType, StructType, StructField
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import count, when, col, isnan, lit
from pyspark.sql import SparkSession
# PySpark ML libraries


from pyspark.ml.feature import StringIndexer, IndexToString, OneHotEncoder, VectorAssembler, BucketedRandomProjectionLSH, VectorIndexer

# Importing the Correlation module 
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class PreProcess:

  def __init__(self, df_param):

    self.sdf = df_param
    self.columns_types_atribute = self.columns_types(self.sdf)

    self.indexer_models = {}
    self.encoded_models = {}


  def show_summary_general(self, df, n_rows=5):
    """
    Prints the schema and top n rows of a Spark DataFrame to the console. 

    :param df: Spark DataFrame to process
    :param n_rows: Number of top rows to print, defaults to 5
    """
    print("Dataframe schema: \n")
    df.printSchema()
    print("\n")

    print(f"Showing top {n_rows} rows: \n")
    df.show(n_rows)
    print("\n")


  def show_summary_categoricals(self, df):
    """
    Prints the value counts of each categorical column of a Spark DataFrame to the console.
    If the DataFrame does not contain any categorical columns, no data is printed.

    :param df: Spark Dataframe to process
    """
    preprocess_instance = PreProcess(df)
    categorical_cols = preprocess_instance.columns_types_atribute['categorical_cols']

    if categorical_cols:
        print("Showing value counts of each categorical feature: \n")
        for c in categorical_cols:
            df.groupBy(c).count().show()
        print("\n")
    else:
        print("No categorical columns were found in the dataframe")


  def show_summary_numerical(self, df):
    """
    Computes and prints the count, mean, stddev, min, max, and quantiles of each numerical column of a Spark DataFrame to the console.
    If the DataFrame does not contain any numerical columns, no data is printed.

    :param df: Spark Dataframe to process
    """
    preprocess_instance = PreProcess(df)
    numerical_cols = preprocess_instance.columns_types_atribute['numeric_cols']

    if numerical_cols:
        print("Showing overall statistics of each numerical feature: \n")
        df.select(numerical_cols).summary().show()
        print("\n")
    else:
        print("No numerical columns were found in the dataframe")

    
  def show_summary_labels(self, df, labels_col):
    """
    Prints the counts of each class of a classification dataset stored as a Spark DataFrame.

    :param df: Spark Dataframe to process
    :param labels_col: Name of the DataFrame column that hold the classes
    """
    print("Showing dataset class distribution: \n")
    df.groupBy(labels_col).count().show()
    print("\n")


  def show_summary_nulls(self, df, spark_session):
    """
    Computes the number of missing values (such as NULL, NaN, empty strings, etc.) in a Spark DataFrame and prints it to the console.

    :param df: Spark Dataframe to process
    :param spark_session: Spark session instance that will be used to process the dataframe.
    """
    preprocess_instance = PreProcess(df)

    # Null counts and percentages per column
    print("Showing missing value counts and percentages per column: \n")
    null_counts = preprocess_instance.get_missing_values_count(df, spark_session)
    null_counts.show()
    print("\n")


  def stratified_train_test_split(self, df, labels_col, train_ratio, seed):
    train_dfs = []
    test_dfs = []
    for label in df.select(labels_col).distinct().collect():
        df_label = df.filter(df[labels_col] == label[0])

        df_train, df_test = df_label.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

        train_dfs.append(df_train)
        test_dfs.append(df_test)
    
    train_df_union = train_dfs[0]
    test_df_union = test_dfs[0]
    for i in range(1, len(train_dfs)):
        train_df_union = train_df_union.union(train_dfs[i])
        test_df_union = test_df_union.union(test_dfs[i])

    return train_df_union, test_df_union


  def preprocess_data(self, df, labels_col, clean_nulls_options, transformation_categorical, transformation_numerical, stratified_split, train_split_ratio, resampling, resampling_options=None, seed=None):
    preprocess_instance = PreProcess(df)
    categorical_cols = preprocess_instance.columns_types_atribute['categorical_cols']
    numerical_cols = preprocess_instance.columns_types_atribute['numeric_cols']

    # Step 1: Clean null values
    df = preprocess_instance.clean_nulls(df, **clean_nulls_options)

    # Step 2: Clean duplicate rows
    df = df.dropDuplicates()

    # Step 3: Process labels column
    if labels_col in categorical_cols:
        categorical_cols.remove(labels_col)
        labels_column_mapping = [(labels_col, labels_col + "_index")]

        df = preprocess_instance.string_index_columns(df, labels_column_mapping)
        
        df = df.drop(labels_col)
    elif labels_col in numerical_cols:
        numerical_cols.remove(labels_col)

    # Step 4: Process categorical columns
    if categorical_cols:
        match transformation_categorical:
            case 'index':
                index_column_mapping = [(col, col + "_index") for col in categorical_cols]
                df = preprocess_instance.string_index_columns(df, index_column_mapping)

                df = df.drop(*categorical_cols)

            case 'encode':
                index_column_mapping = [(col, col + "_index") for col in categorical_cols]
                df = preprocess_instance.string_index_columns(df, index_column_mapping)

                encode_column_mapping = [(col + "_index", col + "_encoded") for col in categorical_cols]
                df = preprocess_instance.encoded_index_columns(df, encode_column_mapping)

                df = df.drop(*[col + "_index" for col in categorical_cols])
                df = df.drop(*categorical_cols)

    # Step 5: Process numerical columns
    if numerical_cols:
        match transformation_numerical:
            case 'normalize':
                df = preprocess_instance.normalize_min_max_values(numerical_cols, df)
                df = df.drop(*numerical_cols)

    # Step 6: Vectorize features
    feature_cols = df.columns
    feature_cols.remove(labels_col)
    df = preprocess_instance.vector_feature_column(df, feature_cols, outputFeatureCols='features')
    df = df.select(['features', labels_col])

    # Step 6: Perform train/test split
    if stratified_split:
        train_df, test_df = self.stratified_train_test_split(df, labels_col, train_split_ratio, seed)
    else:
        train_df, test_df = df.randomSplit([train_split_ratio, 1-train_split_ratio], seed=seed)

    # Step 5: Perform resampling techniques
    match resampling:
        case 'undersample_random':
            train_df = preprocess_instance.undersample_random(train_df, labels_col)
        case 'oversample_random':
            train_df = preprocess_instance.oversample_random(train_df, labels_col)
        case 'nearmiss_v2':
            train_df = preprocess_instance.undersample_nearmiss_v2(train_df, 'features', labels_col, **resampling_options)
        case 'class_weights':
            train_df = preprocess_instance.compute_class_weights_column(df, labels_col)

    return train_df, test_df


  def columns_types(self, sdf):
      """
      Identify the types of columns in a Spark DataFrame.

      Args:
      self: Reference to the class instance.
      sdf (DataFrame): The input Spark DataFrame.

      Returns:
      dict: A dictionary containing lists of column names grouped by their data types.
            Keys: "numeric_cols", "categorical_cols", "boolean_cols", "array_cols", "struct_cols", "unknown_cols"
      """
      # Identify columns by their data types
      numeric_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType, LongType))]
      categorical_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, StringType)]
      boolean_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, BooleanType)]
      array_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, ArrayType)]
      struct_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, StructType)]
      unknown_cols = [col for col in sdf.columns if col not in numeric_cols and col not in categorical_cols and col not in boolean_cols and col not in array_cols and col not in struct_cols]

      # Create a dictionary to store column types
      columns_types_dict = {
          "numeric_cols": numeric_cols,
          "categorical_cols": categorical_cols,
          "boolean_cols": boolean_cols,
          "array_cols": array_cols,
          "struct_cols": struct_cols,
          "unknown_cols": unknown_cols,
      }
      
      return columns_types_dict
  
  def get_missing_values_count(self, df : DataFrame, spark_session : SparkSession):
    """
      Count the number of None, "NULL", "", Null and NaN values in each column of the dataframe

       param df: Spark DataFrame to process.
       param spark_session is the spark sesiion used to process the dataframe.
       return: Spark DataFrame with four columns, Column, Type Count and Percentage, where Count is greater than 0.
          
    """
      # Define the types we want to check
    types_to_check = [
        ("None", lambda c: col(c).contains('None')),
        ("NULL_string", lambda c: col(c).contains('NULL')),
        ("Empty_string", lambda c: col(c) == ''),
        ("Null", lambda c: col(c).isNull()),
        ("NaN", lambda c: isnan(col(c)))
    ]

    # Create a list to save our count expressions
    count_expressions = []

    # For each column and each type, create a count expression
    for c in df.columns:
        for type_name, condition in types_to_check:
            count_expressions.append(
                count(when(condition(c), c)).alias(f"{c}_{type_name}")
            )

    # Create the result DataFrame
    result_df = df.select(count_expressions)

    # Create a schema for our final DataFrame
    schema = StructType([
        StructField("Column", StringType(), False),
        StructField("Type", StringType(), False),
        StructField("Count", LongType(), False),
        StructField("Percentage", FloatType(), False)
    ])

    # Create a list to hold our rows
    rows = []

    # For each column and type, extract the count and create a row
    for c in df.columns:
        for type_name, _ in types_to_check:
            count_value = result_df.select(f"{c}_{type_name}").collect()[0][0]
            rows.append((c, type_name, count_value,(count_value*100)/df.select(count(c)).collect()[0][0]))

    # Create the final DataFrame
    missing_values_count = spark_session.createDataFrame(rows, schema)

    # Show the result
    missing_values_count = missing_values_count.filter(missing_values_count.Count > 0).show()
    return missing_values_count
  
  
  def get_null_counts(self, df: DataFrame, with_percentages=False):
    """
    Counts the number of nulls of each column of a spark Dataframe and returns the counts in a separate Dataframe.
    If with_percentages is set to True, also returns a Dataframe with the percentage of nulls in each column.

    :param df: Spark DataFrame to process.
    :param with_percentages: If set to True, the null percentages per column will also be calculated and returned
    in an additional Dataframe

    :return: Spark Dataframe with calculated null counts per column, or tuple of Spark Dataframes, where one Dataframe
    will hold the column null counts and the other will hold the column null percentages.
    """
    null_counts = df.select([
        F.sum(
            F.when(F.col(column).isNull(), 1)\
            .otherwise(0)
        ).alias(column + "_nulls")
        for column in df.columns
    ])

    if with_percentages:
        total_count = df.count()
        null_percentages = null_counts.select([
            (F.col(column) / total_count).alias(column + "%")
            for column in null_counts.columns
        ])

        return null_counts, null_percentages
    
    else:
        return null_counts
    

  def clean_nulls(self, df: DataFrame, row_threshold: float, column_threshold: float, default_values: dict, unaccounted_nulls_behavior='drop_rows'):
    """
    Clean up null values in a Spark Dataframe according to a specified criteria.
    This method can eliminate nulls by dropping columns, rows, or filling them up with a specific value.

    :param df: Spark DataFrame to process.
    :param row_threshold: Float value indicating the maximum proportion of null values that will be tolerated in a row. All rows with
    a higher proportion of nulls will be dropped.
    :param column_threshold: Float value indicating the maximum proportion of null values that will be tolerated in a coumn. All columns
    with a higher proportion of nulls will be dropped.
    :param default_values: Dictionary that specifies a mapping from column name (key) and desired fill value. The fill value can also be
    a Spark aggregate function in order to fill null values with the corresponding calculation. Currently, the supported aggregate
    functions are ('sum','count','avg','min', 'max', and 'median'). After dropping all rows and columns according to threshold rules, 
    the remaining null values will be filled according to this mapping.
    :param unnacounted_nulls_behavior: String, can be 'drop_rows' or 'drop_cols'. Specifies what action to take if null values are still
    present in the Dataframe after all the previous rules have been applied. If 'drop_rows', all remaining rows with null values will
    be dropped. If 'drop_cols', all remaining columns with null values will be dropped.

    :return: Spark Dataframe with cleaned up null values.
    """
    SUPPORTED_AGGREGATE_FUNCTIONS = ('sum','count','avg','min','max','median')
    
    assert unaccounted_nulls_behavior in ('drop_rows', 'drop_cols'), "Parameter unnacounted_nulls behavior must be in ('drop_rows', 'drop_cols')"

    total_count = df.count()

    drop_cols = [
        column for column in df.columns if (df.filter(F.col(column).isNull()).count() / total_count > column_threshold)
    ]
    df = df.drop(*drop_cols)

    row_threshold_absolute = int(len(df.columns) * (1 - row_threshold))
    df = df.dropna(thresh=row_threshold_absolute)

    null_cols_updated = [
        column for column in df.columns if (df.filter(F.col(column).isNull()).count() > 0)
    ]
    default_values_updated = {column: default_values[column] for column in null_cols_updated if column in default_values.keys()}
    for column, fill_value in default_values_updated.items():
        if fill_value in SUPPORTED_AGGREGATE_FUNCTIONS:
            fill_value_agg = df.select(column).agg({column: fill_value}).first()[0]
            df = df.fillna({column: fill_value_agg})
        else:
            df = df.fillna({column: fill_value})

    null_cols_unnacounted = [
        column for column in df.columns if (df.filter(F.col(column).isNull()).count() > 0)
    ]
    if null_cols_unnacounted:
        if unaccounted_nulls_behavior == 'drop_rows':
            df = df.dropna(how='any')
        else:
            df = df.drop(*null_cols_unnacounted)
    
    return df
  

  def undersample_random(self, df: DataFrame, labels_column: str) -> DataFrame:
    """
    Performs random under-sampling of the majority class on imbalanced datasets. 
    For multiclass classification problems, all classes but the minority class get undersampled. 
    Each class get undersampled to approximately (but not precisely) the same proportion as the minority class.

    :param df: Spark DataFrame to process.
    :param labels_column: Name of the DataFrame column that holds the label values
    :return: Spark Dataframe with the majority classes randomly undersampled.
    """
    label_counts = df.groupby(labels_column).count()

    label_counts_min = label_counts.agg({"count":"min"}).collect()[0][0]

    union_df = None

    for row in label_counts.collect():
        label = row[labels_column]
        count = row['count']

        if count > label_counts_min:
            label_df = df.filter(col(labels_column) == label)
            ratio = label_counts_min / count

            processed_df = label_df.sample(withReplacement=False, fraction=ratio)
        else:
            processed_df = df.filter(col(labels_column) == label)

        if not union_df:
            union_df = processed_df
        else:
            union_df = union_df.union(processed_df)
    
    return union_df
  

  def undersample_nearmiss_v2(self, df: DataFrame, features_column: str, labels_column: str, k: int) -> DataFrame:
    """
    Performs under-sampling of the majority class on imbalanced datasets using the NEARMISS-2 algorithm. 
    Currently, this implementation is only for binary classification problems. The majority class will be
    under-sampled to have the same amount of data that the minority class

    :param df: Spark DataFrame to process.
    :param features_column: Name of the DataFrame column that holds the feature vectors
    :param labels_column: Name of the DataFrame column that holds the label values
    :param k: Number of neighbors to consider on the nearest neighbors-based heuristic
    :return: Spark Dataframe with the majority class undersampled.
    """
    # Get data counts for each label and identify the minority class
    label_counts = df.groupby(labels_column).count()

    majority_count = label_counts.orderBy(F.desc(F.col('count'))).first()['count']

    minority_label = label_counts.orderBy(col('count')).first()[labels_column]
    minority_count = label_counts.orderBy(col('count')).first()['count']

    # Assign unique ids to each minority and majority sample to facilitate calculations
    df_minority = df.filter(col(labels_column) == minority_label).withColumn("id", F.monotonically_increasing_id())
    df_majority = df.filter(col(labels_column) != minority_label).withColumn("id", F.monotonically_increasing_id())

    # Fit a BucketRandomProjectionLSH model to calculate approximate euclidean distances
    features_len = len(df.first()[features_column])
    bucket_length = pow(majority_count, -1/features_len)

    brp = BucketedRandomProjectionLSH(inputCol=features_column, outputCol="hashes", bucketLength=bucket_length, numHashTables=3)
    model = brp.fit(df_majority)

    df_distances = model.approxSimilarityJoin(df_majority, df_minority, float("inf"), distCol="distance")

    # For each majority sample, only consider the k minority samples with the greatest distance
    window_spec = Window.partitionBy("datasetA.id").orderBy(F.desc("distance"))
    df_distances = df_distances.withColumn("distance_rank", F.row_number().over(window_spec))
    df_distances = df_distances.filter(col("distance_rank") <= k)
    # For each majority sample, compute the average distance to the k minority samples that were previously selected
    df_distances = df_distances.groupBy("datasetA.id", "datasetA."+features_column, "datasetA."+labels_column).agg({"distance": "avg"})
    df_distances = df_distances.withColumnRenamed("avg(distance)", "avg_distance")

    # Select the majority samples with the smallest average distance
    df_majority_undersampled = df_distances.orderBy("avg_distance").limit(minority_count).select(features_column, labels_column)

    return df_minority.select(features_column, labels_column).union(df_majority_undersampled)
  

  def oversample_random(self, df: DataFrame, labels_column: str):
    """
    Performs random over-sampling of the minority classes on imbalanced datasets. 
    For multiclass classification problems, all classes but the majority class get oversampled. 
    Each class get oversampled to approximately (but not precisely) the same proportion as the majority class.

    :param df: Spark DataFrame to process.
    :param labels_column: Name of the DataFrame column that holds the label values
    :return: Spark Dataframe with the minority classes randomly oversampled.
    """
    label_counts = df.groupby(labels_column).count()

    label_counts_max = label_counts.agg({"count":"max"}).collect()[0][0]

    union_df = None

    for row in label_counts.collect():
        label = row[labels_column]
        count = row['count']

        if count < label_counts_max:
            label_df = df.filter(col(labels_column) == label)
            ratio = label_counts_max / count

            processed_df = label_df.sample(withReplacement=True, fraction=ratio)
        else:
            processed_df = df.filter(col(labels_column) == label)

        if not union_df:
            union_df = processed_df
        else:
            union_df = union_df.union(processed_df)
    
    return union_df


  def string_index_columns(self, df: DataFrame, columns_to_string_index: list[tuple[str]]) -> DataFrame:
        
        """
         Indexes the specified columns of a DataFrame.

         :param df: Spark DataFrame to process.
         :param columns_to_string_index: List of tuples with the names of the columns to be indexed and the names of the new indexed columns.
         :return: DataFrame with the additional columns indexed.
        """

        for col_name, new_col_name in columns_to_string_index:
            indexer =StringIndexer(inputCol=col_name, outputCol=new_col_name).fit(df)
            df = indexer.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
            self.indexer_models[new_col_name] = indexer
        return df
  
  def encoded_index_columns(self, df: DataFrame, columns_to_encode: list[tuple[str]]) -> DataFrame:
        
        """
         Encode the specified columns of a DataFrame.

         :param df: Spark DataFrame to process.
         :param columns_to_encode: List of tuples with the names of the columns to be encoded and the names of the new encoded columns.
         :return: DataFrame with the additional columns indexed.
        """

        for col_name, new_col_name in columns_to_encode:
            encoder =OneHotEncoder(inputCol=col_name, outputCol=new_col_name).fit(df)
            df = encoder.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
            self.encoded_models[new_col_name] = encoder
        return df
  def vector_feature_column(self, df: DataFrame, feature_cols: list[str], outputFeatureCols: str) -> DataFrame:
        
        """
         Vectorize the specified columns of a DataFrame.

         param df: Spark DataFrame to process.
         param columns_to_create_feature_vector: List of names of the columns to be added to features vector.
         param outputFeatureCols: name of the new column.
         return: DataFrame with the additional vector column.
        """

        # Create a VectorAssembler object
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol=outputFeatureCols, handleInvalid="skip")

        df = vector_assembler.transform(df)
        return df
  def boolean_index_columns(self, df: DataFrame, columns_to_boolean_index: list[tuple[str]]) -> DataFrame:
        
        """
         Convert the specified boolean columns of a DataFrame into integers.

         param df: Spark DataFrame to process.
         param columns_to_boolean_index: List of tuples with the names of the boolean columns to be converted and the names of the new integer  columns.
         return: DataFrame with the additional columns indexed.
        """

        for col_name, new_col_name in columns_to_boolean_index:
            if isinstance(df.schema[col_name].dataType,BooleanType):
                df = df.withColumn(new_col_name,when(col(col_name)==True,1).otherwise(0))
            else:
                raise ValueError(f'The dataType of the column {col_name} is not booleanType(), is {str(df.schema[col_name].dataType)}')
  def revert_string_index(self, df: DataFrame, columns_to_revert: list) -> DataFrame:
        
        """
         Apply inverse operation to revert the indexes to the original strings for the specified columns.

         :param df: Spark DataFrame to process.
         :param columns_to_revert: List of tuples with the names of the indexed columns and the names of the new columns of reverted strings.
         :return: DataFrame with the additional columns of strings reverted.
        """
        for indexed_col_name, reverted_col_name in columns_to_revert:
            # Get labels of the corresponding StringIndexerModel model
            labels = self.indexer_models[indexed_col_name].labels
            # Create and apply the IndexToString object
            converter = IndexToString(inputCol=indexed_col_name, outputCol=reverted_col_name, labels=labels)
            df = converter.transform(df)
        return df
  

  def revert_item_index_in_recommendations(self, df, recommendations_col='recommendations', item_index_col='productIndex'):
      """
      Revert the item index in the recommendations column back to the original item IDs using StringIndexer labels.

      Args:
      df (DataFrame): The input DataFrame containing recommendation data.
      recommendations_col (str): The name of the column containing the recommendations. Default is 'recommendations'.
      item_index_col (str): The name of the column containing the item indices. Default is 'productIndex'.

      Returns:
      DataFrame: The input DataFrame with the recommendations column reverted to contain original item IDs.
      """
      # Get the StringIndexer model labels for asin/productIndex
      asin_labels = self.indexer_models[item_index_col].labels
      
      # Define a UDF to map product indices to asin
      def map_index_to_asin(recommendations):
          return [(asin_labels[index], score) for index, score in recommendations]
      
      map_index_to_asin_udf = udf(map_index_to_asin, ArrayType(StructType([
          StructField("asin", StringType(), nullable=False),
          StructField("score", FloatType(), nullable=False)
      ])))


      # Apply the UDF to the recommendations column to transform the indexes to asin
      
      df = df.withColumn(recommendations_col, map_index_to_asin_udf(col(recommendations_col)))
      return df


  def drop_null_values(self):

    # Copy of the dataframe
    df_copy = self.sdf.alias("df_copy")

    # Drop rows with any null values in any column
    df_no_null = df_copy.na.drop()

    return df_no_null

  def normalize_min_max_values(self, features_to_normalize: list, sdf):
      """
      Normalize the specified features in the Spark DataFrame using min-max scaling.

      Args:
      self: Reference to the class instance.
      features_to_normalize (list): A list of feature names to be normalized.
      sdf (DataFrame): The input Spark DataFrame containing the features.

      Returns:
      DataFrame: The input DataFrame with the specified features normalized.
      """
      # Copy of the dataframe
      #df_copy = self.sdf.alias("df_copy")

      feature_scaled = []
      for feature in features_to_normalize:
          # Compute the min and max values for the current feature
          min_max_values = sdf.agg(F.min(feature).alias("min_" + feature),
                                  F.max(feature).alias("max_" + feature)).collect()  # list type

          min_feature = min_max_values[0]["min_" + feature]  # numeric value
          max_feature = min_max_values[0]["max_" + feature]  # numeric value

          # Apply min-max normalization
          df_copy = sdf.withColumn(feature + "_normalized",
                                  (F.col(feature) - min_feature) / (max_feature - min_feature))

      return df_copy
  

  def compute_class_weights_column(self, df: DataFrame, labels_column: str) -> DataFrame:
      """
      Calculate class weights for unbalanced datasets and put them in a new class_weights column

      :param df: Spark DataFrame to process
      :param labels_column: Name of the DataFrame column that holds the label values
      :return: Spark DataFrame with an additional class_weights column that holds the class weights associated with each row
      """
      label_counts = df.groupby(labels_column).count().collect()
      total_count = df.count()

      n_labels = len(label_counts)

      class_weights = {}

      for row in label_counts:
          class_weights[row[labels_column]] = total_count / (n_labels * row['count'])


      def get_class_weight(label):
          return float(class_weights[label])
      
      get_class_weight_udf = udf(get_class_weight, FloatType())

      return df.withColumn("class_weights", get_class_weight_udf(col(labels_column)))
  
  def churn(self, sdf, churn:str):
      logger = logging.getLogger(__name__)
      logging.basicConfig( encoding='utf-8', level=logging.INFO)
      """
      Identify the types of columns in a Spark DataFrame.

      Args:
      self: Reference to the class instance.
      sdf (DataFrame): The input Spark DataFrame.
      churn: The name of the churn column on the input Spark DataFrame.

      Returns:
      sdf: A Spark DataFrame with the churn column pre processed depending on its data type.
      """
      
      # Identify columns by their data types
      numeric_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType, LongType))]
      categorical_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, StringType)]
      boolean_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, BooleanType)]
      array_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, ArrayType)]
      struct_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, StructType)]
      unknown_cols = [col for col in sdf.columns if col not in numeric_cols and col not in categorical_cols and col not in boolean_cols and col not in array_cols and col not in struct_cols]

      # Create a dictionary to store column types
      columns_types_dict = {
          "numeric_cols": numeric_cols,
          "categorical_cols": categorical_cols,
          "boolean_cols": boolean_cols,
          "array_cols": array_cols,
          "struct_cols": struct_cols,
          "unknown_cols": unknown_cols,
      }
      
      churn1 = churn

      found_churn1 = False

      for key, value in columns_types_dict.items():
        if isinstance(value, list):
        # Iterate over list elements to check if column is present in list item
          for item in value:
            if isinstance(item, str) and item == churn1:
                found_churn1 = True
                # Column processing according to its data type
                if key == "numeric_cols":
                      indexer =StringIndexer(inputCol=churn1, outputCol="indexed_"+ churn1).fit(sdf)
                      sdf = indexer.transform(sdf)
                     # Store model StringIndexerModel in dict f or revert_string_index method
                      self.indexer_models["indexed_"+ churn1] = indexer
                     ## cambiar prints por login
                      logger.info(f"Se encontró '{churn1}' en la lista de '{key}' , se aplicó string indexer")
                elif key == "boolean_cols":  
                    integer_column = when(col(churn1) == False, 0).otherwise(1)
                    sdf = sdf.withColumn(churn1 + '_booleantointeger', integer_column) 
                    logger.info(f"Se encontró '{churn1}' en la lista de '{key}' , se aplicó boolean to integer") 
                elif key == "categorical_cols":
                    stringIndexer = StringIndexer(inputCol= churn1, outputCol="indexed_"+ churn1)
                    sdf = stringIndexer.fit(sdf).transform(sdf)
                    sdf.show()
                    logger.info(f"Se encontró '{churn1}' en la lista de '{key}', se aplicó string indexer")
                break  
          else:
            continue  
          break 
      
      if not found_churn1:
       logger.info(f"'{churn1}' no es una columna del dataframe")
   
      return sdf

  def churn_inverter(self, sdf, churn:str):
      """
      Inverts the churn values of  the churn column in a Spark DataFrame.

      Args:
      self: Reference to the class instance.
      sdf (DataFrame): The input Spark DataFrame.
      churn: The name of the churn column on the input Spark DataFrame.

      Returns:
      sdf: A Spark DataFrame with the churn column vlaues inverted.
      """
      churn1 = churn   
      integer_column = when(col(churn1) == 0, 1).otherwise(0)
      sdf = sdf.withColumn(churn1 + '_inverted', integer_column) 
      sdf = sdf.drop(churn1)
      return sdf
  
  
  def get_indexed_and_encode_vectors(self,df,feature_cols : list, label_col:str):
     """
      Preprocess data to use it in RandomForestCassifier or GBTclassifier

       param df: Spark DataFrame to process
       param feature_cols: list with the names of cols that will be used as features, 
       param label_col: name of the DataFrame column that holds the label values
       return: A list with the following output
              index 0: Spark DataFrame with additional indexed, encoded and vectorized cols, and class_weights column that holds the class weights associated with each row
              index 1: name of vector column of indexed features (can be used to train the model)
              index 2: name of vector column of encoded features (can be used to train the model)
              index 3: new name for churn column after processing
              index 4 : name for column with the weight of classes
               
     """
     #Identifying numeric, categorical and boolean columns that exists in both, df and feature cols
     numeric_cols = [c for c in self.columns_types(df)["numeric_cols"] if c in feature_cols]
     categorical_cols = [c for c in  self.columns_types(df)["categorical_cols"] if c in feature_cols]
     boolean_cols =[c for c in self.columns_types(df)["boolean_cols"] if c in feature_cols]
    #Defining default values 
     categorical_cols_indexed =  []
     columns_to_string_index=[]
     boolean_cols_indexed =  []
     columns_to_boolean_index = []
     categorical_cols_encoded = []
     boolean_cols_encoded = []
     columns_indexed = []
     columns_encoded = []
     columns_to_encode = []
     maxCategories = 5
     weigth_class_column = "class_weights"
     new_label_col = label_col
     if len(categorical_cols) > 0:
         #Creating the list of tuples with categorical cols to be indexed and encoded
         categorical_cols_indexed =  [f"{c}Index" for c in categorical_cols]
         columns_to_string_index=list(zip(categorical_cols,categorical_cols_indexed))
         categorical_cols_encoded = [f"{c}Encoded" for c in categorical_cols_indexed]
         # Set maxCategories as the maximum number of categories in every categorical and boolean feature
         distinct_counts = df.agg(*[countDistinct(c).alias(c) for c in categorical_cols+boolean_cols])
         maxCategories = max([distinct_counts.select(c).collect()[0][c] for c in distinct_counts.columns])
     if len(boolean_cols) > 0:
         #Creating the list of tuples with boolean cols to be indexed and encoded
         boolean_cols_indexed =  [f"{c}Index" for c in boolean_cols]
         columns_to_boolean_index = list(zip(boolean_cols,boolean_cols_indexed))
         boolean_cols_encoded = [f"{c}Encoded" for c in boolean_cols_indexed]
     if len(boolean_cols) > 0 or len(categorical_cols):   
         #creating the list of tuples with boolean and categorical cols to be encoded     
         columns_indexed = categorical_cols_indexed + boolean_cols_indexed
         columns_encoded = categorical_cols_encoded + boolean_cols_encoded
         columns_to_encode = list(zip(columns_indexed,columns_encoded))



      #Defining two new lists of features with the  features that just were indexed and the features that wre encoded after indexed, respectively
     indexed_features_col_to_train_model =  categorical_cols_indexed + boolean_cols_indexed + numeric_cols
     encoded_features_col_to_train_model =  columns_encoded + numeric_cols
       
    
      #Indexing the label column according to its type
    
     if isinstance(df.schema[label_col].dataType,StringType):
        df= self.string_index_columns(df, [(label_col,"Churn_Index")])
        new_label_col = "Churn_Index"
     elif isinstance(df.schema[label_col].dataType,BooleanType):
        df = self.boolean_index_columns(df,[(label_col,"Churn_Index")])
        new_label_col = "Churn_Index"
     elif label_col in  self.columns_types(df)["numeric_cols"]:
        pass
     else:
        raise ValueError(f"Your label column {label_col} is not a supported type for this method and maybe requires another kind of treatment.")
    
     if len(categorical_cols) > 0:
        #Indexing categorical columns from features 
        df= self.string_index_columns(df, columns_to_string_index)
     if len(boolean_cols):
        #Indexing boolean columns from features 
        df = self.boolean_index_columns(df,columns_to_boolean_index)
     if len(boolean_cols) > 0 or len(categorical_cols):   
        #Encoding indexed categorical and boolean columns from features
        df = self.encoded_index_columns(df, columns_to_encode)
     #Adding column with the weight classes to help with imbalanced classes
     df = self.compute_class_weights_column( df, new_label_col)

     #merging columns in indexed_features_col_to_train_model into a vector column
     df = self.vector_feature_column(df, indexed_features_col_to_train_model,"IndexedFeatures") 

     #merging columns in encoded_features_col_to_train_model into a vector column
     df = self.vector_feature_column(df, encoded_features_col_to_train_model,"IndexedEncodedFeatures") 
    
    
    
     # Automatically identify categorical features from features that just were indexed, and index them.
     # Set maxCategories so features with > maxCategories distinct values are treated as continuous.
     vi = VectorIndexer(inputCol="IndexedFeatures", outputCol="VectorIndexedFeatures", maxCategories=maxCategories).fit(df)
     df = vi.transform(df)
    # Automatically identify categorical features from features that were encoded after indexation, and index them.
    # Set maxCategories so features with > 6 distinct values are treated as continuous.
     vi = VectorIndexer(inputCol="IndexedEncodedFeatures", outputCol="VectorIndexedEncodedFeatures", maxCategories=maxCategories).fit(df)
     df = vi.transform(df)
     return [df,"VectorIndexedFeatures","VectorIndexedEncodedFeatures",new_label_col,weigth_class_column] 

  def show_correlation_matrix(self,df : DataFrame, label_col: str,correlation_method : str):
        """
        Preprocess data to use it in RandomForestCassifier or GBTclassifier

        param df: Spark DataFrame to process
        param label_col: name of the DataFrame column that holds the label values
        param correlation_method: name of the correlation to be calculated, "spearman" or "pearson"
        return: show a plot of the correlation matrix in a coolwarm palette
        """
        columns = df.columns
        data_to_plot = self.get_indexed_and_encode_vectors(df,columns , label_col)
        df= data_to_plot[0]
        VectorIndexedFeatures = data_to_plot[1]
        # Calculate correlation matrix
        if correlation_method in ["spearman","pearson"]:
            pearson_corr_matrix = Correlation.corr(df, VectorIndexedFeatures, method= correlation_method).head()
        else:
            raise ValueError("Not supported method for correlation.")
        correlation_matrix = pearson_corr_matrix[0].toArray()
        # Convert to Pandas DataFrame
        df2 = pd.DataFrame(correlation_matrix,index = columns, columns= columns)
        # Set up the matplotlib figure
        plt.figure(figsize=(20, 20))

        # Create the heatmap
        sns.heatmap(df2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

        # Set the title
        plt.title('Correlation Matrix Heatmap')

        # Show the plot
        plt.show()

  @classmethod
  def load_data(cls, file_path_df: str, spark):
    """
    Load data from CSV, Parquet, or JSON files into a Spark DataFrame.
    
    Parameters:
    - file_path_df (str): The path to the data file.
    - spark: The Spark session instance.
    
    Returns:
    - A Spark DataFrame containing the loaded data.
    
    Raises:
    - ValueError: If the file format is not supported (i.e., not CSV, Parquet, or JSON).
    """
    if file_path_df.endswith('.csv'):
        df = spark.read.csv(file_path_df, header = True,inferSchema=True)
    elif file_path_df.endswith('.parquet'):
        df = spark.read.parquet(file_path_df)
    elif file_path_df.endswith('.json'):
        df = spark.read.json(file_path_df)
    else:
        raise ValueError("Unsupported file format. Only CSV, Parquet, and JSON files are supported.")
    return df  