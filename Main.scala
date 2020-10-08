import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.sql.functions._


object Parameters {
  // Execution parameters
  val training_mode = false
  val dataset = "StanfordSentimentTreebank"
  val train_split = 0.9
  val seed = 42
}

object Main {
  
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
    delete_output("outputRandomForest/", spark)
    delete_output("logisticRegressionCheckpoint/", spark)
    delete_output("randomForestCheckpoint/", spark)
    delete_output("singleOutput/", spark)
    
    // Differentiate between training mode and prediction mode (streaming)
    if (Parameters.training_mode) {
      training()
    } else {
      streaming(spark)
    }
  }

  // Function for training the models
  def training(): Unit = {
    // Read the data
    val data = Reader.read_dataset(Parameters.dataset)

    // Perform train-test split
    var Array(train, test) = data.randomSplit(Array(Parameters.train_split, 1 - Parameters.train_split), seed = Parameters.seed)

    // Apply preprocessing to both sets
    train = Preprocess.filter(train)
    test = Preprocess.filter(test)

    // Extract features
    val W2V = Models.train_w2v(train)
    train = W2V.transform(train)
    test = W2V.transform(test)

    // Fit and evaluate models
    val log_reg = Models.train_log(train)
    val forest = Models.train_forest(train)
    Models.eval(log_reg, test)
    Models.eval(forest, test)
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
    val W2V = Models.load_word_vector()
    val log_reg = Models.load_regression()
    val forest = Models.load_forest()

    // Extract features out of preprocessed tweet from the stream
    val filtered = W2V.transform(Preprocess.filter(line))

    // Predict the sentiment of the tweet
    val pred1 = log_reg.transform(filtered)
    val pred2 = forest.transform(filtered)

    // Output the predictions
    val query1 = pred1
      .withColumn("timestamp", current_timestamp())
      .select("timestamp", "SentimentText", "prediction")
      .writeStream
      .format("csv")
      .option("checkpointLocation", "logisticRegressionCheckpoint/")
      .option("path", "outputLogisticRegression/")
      .start()

    val query2 = pred2
      .withColumn("timestamp", current_timestamp())
      .select("timestamp", "SentimentText", "prediction")
      .writeStream
      .format("csv")
      .option("checkpointLocation", "logisticRegressionCheckpoint/")
      .option("path", "outputRandomForest/")
      .start()

    //join_outputs(spark)
    query1.awaitTermination()
    query2.awaitTermination()
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
          val df1 = spark.read.option("header","false").csv("outputLogisticRegression/*.csv").toDF("timestamp", "tweet", "LogRegLabel")
          val df2 = spark.read.option("header","false").csv("outputRandomForest/*.csv").as("randForest").select("_c0", "_c2").toDF("timestamp", "RandTreeLabel")
          val df = df1.join(df2, "timestamp")
          df.show(truncate = false)
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
