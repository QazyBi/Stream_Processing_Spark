import org.apache.spark.ml.classification._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.nio.file.{Files, Paths}
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


// class containing 2 employed models (Random forest & Logistic regression)
object Models {

  // setting up model cross-validation and evaluation
  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
  val cross_val: CrossValidator = new CrossValidator()
    .setEvaluator(evaluator.setMetricName("f1"))
    .setNumFolds(2)
    .setParallelism(2)
  
  // function to train random forest model
  def train_forest(train: Dataset[Row]): RandomForestClassificationModel = {
    
    // initializing the classifier
    val name = "Random_Forest"
    val forest = new RandomForestClassifier()
    
    // initializing the hyperparameters (refer to the report for the explanation of their choice)
    val grid = new ParamGridBuilder()
      .addGrid(forest.numTrees, Array(5, 10))
      .addGrid(forest.maxDepth, Array(5, 10))
      .build()

    // choose the best hyperparameters
    val bestModel = cross_val
      .setEstimator(forest)
      .setEstimatorParamMaps(grid)
      .fit(train)
      .bestModel.asInstanceOf[RandomForestClassificationModel]
    bestModel.write.overwrite().save(name)
    bestModel
  }
  
  // function to train logistic regression
  def train_log(train: Dataset[Row]): LogisticRegressionModel = {
    
    // initializing the classifier
    val name = "Logistic_Regression"
    val log_reg = new LogisticRegression()
      .setElasticNetParam(0)
      .setFamily("multinomial")
    
    // initializing the hyperparameters (refer to the report for the explanation of their choice)
    val grid = new ParamGridBuilder()
      .addGrid(log_reg.regParam, Array(0, 0.01, 0.1))
      .addGrid(log_reg.maxIter, Array(50, 100))
      .build()

    // choose the best hyperparameters
    val bestModel = cross_val
      .setEstimator(log_reg)
      .setEstimatorParamMaps(grid)
      .fit(train)
      .bestModel.asInstanceOf[LogisticRegressionModel]
    bestModel.write.overwrite().save(name)
    bestModel
  }
  
  // training word to vector transformation
  def train_w2v(documentDF: DataFrame): Word2VecModel = {
    
    // initialize the model
    var W2V: Word2VecModel = null
    val name = "Word2Vector"
    
    // load the model if already in streaming mode, train otherwisw
    if (Parameters.training_mode==false) {
      W2V = Word2VecModel.load(name)
    }else {
      val word_vec = new Word2Vec()
        .setInputCol("FilteredSentimentText")
        .setOutputCol("features")
        .setVectorSize(50)
        .setMinCount(5)
        .setSeed(Parameters.seed)
      W2V = word_vec.fit(documentDF)
      W2V.write.overwrite().save(name)
    }
    W2V
  }

  // the following three functions used to load trainined models into the main class
  def load_regression(): LogisticRegressionModel = {
    var log_reg: LogisticRegressionModel = null
    val name = "Logistic_Regression"
    log_reg = LogisticRegressionModel.load(name)
    log_reg
  }

  def load_forest(): RandomForestClassificationModel = {
    var forest: RandomForestClassificationModel = null
    val name = "Random_Forest"
    forest = RandomForestClassificationModel.load(name)
    forest
  }
  
  def load_word_vector(): Word2VecModel = {
    var W2V: Word2VecModel = null
    val name = "Word2Vector"
    W2V = Word2VecModel.load(name)
    W2V
  }

  // function to evaluate the classifiers and output metrics
  def eval(classifier: ProbabilisticClassificationModel[Vector, _], test: Dataset[Row]): Unit = {
    val name = classifier.getClass.toString.split("\\.")(5)
    println("Evaluating " + name)
    val predictions = classifier.transform(test).select("prediction", "label")
    predictions.show(50)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println("F1 score = " + f1)
  }
}
