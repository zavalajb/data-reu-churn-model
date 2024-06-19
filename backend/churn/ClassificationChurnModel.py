
# SQL libraries
from pyspark.sql.functions import lit
# ML libraries
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class RecommendationModel:

  def __init__(self, df_param, features:list, objective:str):

    self.data = df_param
    self.features_atribute = features
    self.objective_atribute = objective

    self.train, self.test = self.split_data()

  def split_data(self):
    """
    Split the data in 80% for training and 20% for testing for the features and objective
    """
    return self.data.randomSplit([0.8, 0.2])

  def build_classification_model(self, labelCol: str, featuresCol: str, ratingCol: str):

    # Initialize the classification Model
    randomforest = RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol,seed = 42)
    # Set considered parameter grid
    # Create a ParamGridBuilder
    paramGrid = (ParamGridBuilder()
                .addGrid(randomforest.numTrees, [10, 20, 30])  # Number of trees
                .addGrid(randomforest.maxDepth, [5, 10, 15])   # Maximum depth of the tree
                .addGrid(randomforest.maxBins, [32, 64, 128])  # Number of bins used when discretizing continuous features
                .addGrid(randomforest.featureSubsetStrategy, ['auto', 'sqrt', 'log2'])  # Strategy to select a subset of features
                .build())    # Set evaluator
    modelEvaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="weightedRecall")



    # Create the CrossValidator
    crossval = CrossValidator(estimator=randomforest,
                          estimatorParamMaps=paramGrid,
                          evaluator=modelEvaluator,
                          numFolds=5)  

    return crossval

  def train_classification_model(self, labelCol: str, featuresCol: str):
    
    crossval = self.build_classification_model(labelCol=labelCol, featuresCol=featuresCol)
    cvModel = crossval.fit(self.train)
    return  cvModel.bestModel


  def performance_evaluation(self, labelCol,featuresCol, predictions):


    evaluator = MulticlassClassificationEvaluator(
    labelCol=labelCol, predictionCol=featuresCol, metricName="weightedRecall"
    )

    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy = {accuracy}")

    # Show detailed evaluation metrics
    evaluator.setMetricName("weightedPrecision")
    precision = evaluator.evaluate(predictions)
    print(f"Weighted Precision = {precision}")

    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictions)
    print(f"Weighted Recall = {recall}")

    evaluator.setMetricName("f1")
    f1 = evaluator.evaluate(predictions)
    print(f"F1 Score = {f1}")
    return [accuracy, precision, recall,f1]
  


  

  
