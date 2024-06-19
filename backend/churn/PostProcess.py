from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType
import boto3

class PostProcess:
    def __init__(self, top_n):
        self.top_n = top_n  # Este atributo almacenará el número de recomendaciones top a retornar

    def get_top_recommendations(self, recommendations):
        # Este método ahora utiliza el atributo self.top_n en lugar de recibir top_n como argumento
        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        top_items = [rec[0] for rec in sorted_recommendations[:self.top_n]]
        return top_items

    def get_user_top_recommendations(self, df, user_id, originalReviewerID):
        # UDF para aplicar el método get_top_recommendations a la columna de recomendaciones
        top_recommendations_udf = udf(lambda recommendations: self.get_top_recommendations(recommendations), ArrayType(StringType()))

        # Filtra el DataFrame por el usuario especificado y aplica el UDF
        filtered_df = df.filter(col(originalReviewerID) == user_id).withColumn("top_recommendations", top_recommendations_udf(col("recommendations")))

        # Extrae el resultado como un diccionario
        result_dict = filtered_df.select(originalReviewerID, "top_recommendations").rdd.flatMap(lambda x: [(x[0], x[1])]).collectAsMap()

        # Formatea el resultado
        if result_dict:
            final_dict = {"ReviewerID": list(result_dict.keys())[0], "recommendations": list(result_dict.values())[0]}
        else:
            final_dict = {"ReviewerID": user_id, "recommendations": []}  # En caso de que no haya recomendaciones para el usuario

        return final_dict
    
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

        try:
            if file_path_df.endswith('.csv'):
                df = spark.read.option("header", True).csv(file_path_df)
            elif file_path_df.endswith('.json') or file_path_df.endswith('.json.gz'):
                df = spark.read.json(file_path_df)
            elif file_path_df.endswith('.parquet').repartition(100):
                df = spark.read.parquet(file_path_df)
        except ValueError:
            # Maneja la excepción ZeroDivisionError
            print("Unsupported file format. Only CSV, Parquet, and JSON files are supported.")
  
        return df
    
    def download_data_from_s3_bucket(aws_access_key_id:str, aws_secret_access_key:str, region_name:str, bucket_name:str, file_path_save:str) -> None:

        # Create an S3 client
        s3_client = boto3.client('s3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name)
        
        s3_client.download_file(bucket_name, "training_data/amazon_union_data.parquet", file_path_save)
        
    def upload_model_to_s3_bucket(aws_access_key_id:str, aws_secret_access_key:str, region_name:str, bucket_name:str, file_path_model:str):

        s3_client = boto3.client('s3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name)

        s3_client.upload_file(file_path_model, bucket_name, "als_model")