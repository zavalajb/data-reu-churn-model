# Class for preproccessing the data

# SQL libraries
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, when
from pyspark.ml.feature import StringIndexer
import logging
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, BooleanType, ArrayType, LongType, StructType, StructField
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
         :param columns_to_string_index: List of tuples with the names of the columns to be indexed and the names of the new indexed columns.
         :return: DataFrame with the additional columns indexed.
        """

        for col_name, new_col_name in columns_to_encode:
            encoder =OneHotEncoder(inputCol=col_name, outputCol=new_col_name).fit(df)
            df = encoder.transform(df)
            # Store model StringIndexerModel in dict f or revert_string_index method
            self.encoded_models[new_col_name] = encoder
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