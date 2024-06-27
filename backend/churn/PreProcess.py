# Class for preproccessing the data

import logging
# SQL libraries
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, BooleanType, ArrayType, LongType, StructType, StructField
# PySpark ML libraries


from pyspark.ml.feature import StringIndexer, IndexToString, OneHotEncoder, VectorAssembler, BucketedRandomProjectionLSH

class PreProcess:

  def __init__(self, df_param):

    self.sdf = df_param
    self.columns_types_atribute = self.columns_types(self.sdf)

    self.indexer_models = {}
    self.encoded_models = {}

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
    functions are ('sum','count','avg','min', and 'max'). After dropping all rows and columns according to threshold rules, 
    the remaining null values will be filled according to this mapping.
    :param unnacounted_nulls_behavior: String, can be 'drop_rows' or 'drop_cols'. Specifies what action to take if null values are still
    present in the Dataframe after all the previous rules have been applied. If 'drop_rows', all remaining rows with null values will
    be dropped. If 'drop_cols', all remaining columns with null values will be dropped.

    :return: Spark Dataframe with cleaned up null values.
    """
    SUPPORTED_AGGREGATE_FUNCTIONS = ('sum','count','avg','min','max')
    
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
  def vector_feature_column(self, df: DataFrame, feature_cols: list[str]) -> DataFrame:
        
        """
         Vectorize the specified columns of a DataFrame.

         :param df: Spark DataFrame to process.
         :param columns_to_create_feature_vector: List of names of the columns to be added to features vector.
         :return: DataFrame with the additional vector column.
        """

   # Create a VectorAssembler object
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

        df = vector_assembler.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
        #self.encoded_models[new_col_name] = encoder
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