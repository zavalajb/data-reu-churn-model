
# SQL libraries
from pyspark.sql.functions import lit
# ML libraries
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class ClassificationModel:

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

  def build_classification_model(self, labelCol: str, featuresCol: str, weightCol : str):
    """
      Ths method initialize a Random Forest Classifier, sets up parameter grid and creaes a CrossValidator

         labelCol: name of the label column in the data frame, 
         featuresCol: name of the features col in the data frame, 
         weightCol : name of the weight col
         return: a CrossValidator object that will be used to perfom k-fold coss-validation on a dataset
    """

    # Initialize the classification Model
    randomforest = RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol,weightCol= weightCol,seed = 42)

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

  def train_classification_model(self, labelCol: str, featuresCol: str, weightCol : str):
    """
         Train the model with an specific param grid performing cross validation

         labelCol: name of the label column in the data frame, 
         featuresCol: name of the features col in the data frame, 
         weightCol : name of the weight col
         return: the model that performed best during cross-validation in train dataset
    """  
    crossval = self.build_classification_model(labelCol=labelCol, featuresCol=featuresCol, weightCol=weightCol)
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
  


  

  
