import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.sql.functions._

object Main {
  
  // Execution parameters
  val training_mode = true
  val dataset = "StanfordSentimentTreebank"
  val train_split = 0.9
  val seed = 42
  
  // Function to get rid of output folders before running jobs
  def delete_output(path: String, spark: SparkSession): Unit = {
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val output = new Path(path)
    if (fs.exists(output))
      fs.delete(output, true)
  }
  
  // Main driver function
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    val spark = SparkSession.builder.appName("Streamer")
      .config("spark.master", "local[2]").getOrCreate

    delete_output("outputLogisticRegression/", spark)
    delete_output("logisticRegressionCheckpoint/", spark)
    delete_output("singleOutput/", spark)
    
    // Differentiate between training mode and prediction mode (streaming)
    if (training_mode) {
      streaming(spark)
    } else {
      training()
    }
  }

  // Function for training the models
  def training(): Unit = {
    // Read the data
    val data = Reader.read(dataset)

    // Perform train-test split
    var Array(train, test) = data.randomSplit(Array(train_split, 1 - train_split), seed = seed)

    // Apply preprocessing to both sets
    train = Preprocessing.preprocess(train)
    test = Preprocessing.preprocess(test)

    // Extract features
    val W2V = FeatureExtractor.trainModel(train)
    train = FeatureExtractor.getVector(W2V, train)
    test = FeatureExtractor.getVector(W2V, test)

    // Fit and evaluate logistic regression model
    val log_reg = Classifier.trainLogisticRegression(train)
    Classifier.evaluateModel(log_reg, test)
  }

  // Function for predictions on stream data
  def streaming(spark: SparkSession): Unit = {
    // Connect to the data source
    val line = spark.readStream
      .format("socket")
      .option("host", "10.90.138.32")
      .option("port", 8989)
      .load().toDF("SentimentText")


    // Load necessary models
    val W2V = FeatureExtractor.loadModel()
    val log_reg = Classifier.loadLogRegModel()

    // Extract features out of preprocessed tweet from the stream
    val filtered = W2V.transform(Preprocessing.filterData(line))

    // Predict the sentiment of the tweet
    val pred = log_reg.transform(filtered)

    // Output the predictions
    val query = pred
      .withColumn("timestamp", current_timestamp())
      .select("timestamp", "SentimentText", "prediction")
      .writeStream
      .format("csv")
      .option("checkpointLocation", "logisticRegressionCheckpoint/")
      .option("path", "outputLogisticRegression/")
      .start()

    join_outputs(spark)
    query.awaitTermination()
  }

  def join_outputs(spark: SparkSession): Unit = {
    val res = new Thread(new Runnable {def run(): Unit = {
      // Wait for first data to arrive
      Thread.sleep(30 * 1000)
      while (true) {
        // Sleep for 10 seconds
        Thread.sleep(10 * 1000)
        val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
        val out_log = new Path("outputLogisticRegression/")
        if (fs.exists(out_log)) {
          delete_output("singleOutput/", spark)
          val df = spark.read.option("header","false").csv("outputLogisticRegression/*.csv").toDF("timestamp", "tweet", "LogRegLabel")
          df.coalesce(1)
            .write.format("csv").option("header", "false").save("singleOutput/")

          // WordCount
          import spark.implicits._
          val words = df.toDF().select("tweet").as[String].flatMap(_.split("\\W+"))
          val wordCounts = words.groupBy("value").count()
          println("WordCount output")
          wordCounts.show(10, truncate=false)
          delete_output("singleOutputWordCount/", spark)
          wordCounts.coalesce(1).write.format("csv").option("header","false").save("singleOutputWordCount/")
        }
      }
    }})

    res.start()
    res.join()
  }
}
