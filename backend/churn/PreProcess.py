# Class for preproccessing the data

# SQL libraries
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, BooleanType, ArrayType, LongType, StructType, StructField
from pyspark.ml.classification import GBTClassifier
# PySpark ML libraries


from pyspark.ml.feature import StringIndexer, IndexToString, OneHotEncoder, VectorAssembler

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
         Indexes the specified columns of a DataFrame.

         :param df: Spark DataFrame to process.
         :param columns_to_encode: List of tuples with the names of the columns to be encoded and the names of the new encoded columns.
         :return: DataFrame with the additional columns indexed.
        """

        for col_name, new_col_name in columns_to_encode:
            encoder =OneHotEncoder(inputCol=col_name, outputCol=new_col_name ,handleInvalid="keep").fit(df)
            df = encoder.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
            self.encoded_models[new_col_name] = encoder
        return df
  def vector_feature_column(self, df: DataFrame, feature_cols: list[str], outputFeatureCol : str) -> DataFrame:
        
        """
         Indexes the specified columns of a DataFrame.

         :param df: Spark DataFrame to process.
         :param columns_to_create_feature_vector: List of names of the columns to be added to features vector.
         :return: DataFrame with the additional vector column.
        """

# Create a VectorAssembler object
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol=outputFeatureCol, handleInvalid="skip")

        df = vector_assembler.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
        #self.encoded_models[new_col_name] = encoder
        return df
  
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
        df = spark.read.option("header", True).csv(file_path_df)
    elif file_path_df.endswith('.parquet'):
        df = spark.read.parquet(file_path_df)
    elif file_path_df.endswith('.json'):
        df = spark.read.json(file_path_df)
    else:
        raise ValueError("Unsupported file format. Only CSV, Parquet, and JSON files are supported.")
    return df