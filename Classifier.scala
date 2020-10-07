import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row}

object Classifier {
  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
  val cv: CrossValidator = new CrossValidator()
    .setEvaluator(evaluator.setMetricName("f1"))
    .setNumFolds(2)
    .setParallelism(2)


  def trainLogisticRegression(train: Dataset[Row]): LogisticRegressionModel = {
    val name = "LogisticRegressionModel"
    val mlr = new LogisticRegression()
      .setElasticNetParam(0)
      .setFamily("multinomial")

    val paramGrid = new ParamGridBuilder()
      .addGrid(mlr.regParam, Array(0, 0.01, 0.1))
      .addGrid(mlr.maxIter, Array(70, 80))
      .build()

    // Run cross-validation, and choose the best set of parameters.
    val bestModel = cv
      .setEstimator(mlr)
      .setEstimatorParamMaps(paramGrid)
      .fit(train)
      .bestModel.asInstanceOf[LogisticRegressionModel]
    bestModel.write.overwrite().save(name)
    bestModel
  }

  def trainRandomForest(train: Dataset[Row]): RandomForestClassificationModel = {
    val name = "RandomForestClassificationModel"
    val rf = new RandomForestClassifier()
      .setCacheNodeIds(true)

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20))
      .addGrid(rf.maxDepth, Array(10, 20))
      .build()

    // Run cross-validation, and choose the best set of parameters.
    val bestModel = cv
      .setEstimator(rf)
      .setEstimatorParamMaps(paramGrid)
      .fit(train)
      .bestModel.asInstanceOf[RandomForestClassificationModel]
    bestModel.write.overwrite().save(name)
    bestModel
  }

  def loadLogRegModel(): LogisticRegressionModel = {
    var logRegModel: LogisticRegressionModel = null
    val name = "LogisticRegressionModel"
    try {
      logRegModel = LogisticRegressionModel.load(name)
    } catch {
      case e: Exception => throw new Exception(e + "\nMust pre-train models first, set PRODUCTION = false");
    }
    logRegModel
  }

  def loadRandomForestModel(): RandomForestClassificationModel = {
    var randForestModel: RandomForestClassificationModel = null
    val name = "RandomForestClassificationModel"
    try {
      randForestModel = RandomForestClassificationModel.load(name)
    } catch {
      case e: Exception => throw new Exception(e + "\nMust pre-train models first, set PRODUCTION = false");
    }
    randForestModel
  }

}
