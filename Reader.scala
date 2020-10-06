import org.apache.spark.sql.{DataFrame, SparkSession}

object Reader {
  def read(inputPath: String): DataFrame = {
// Gets an existing SparkSession or creates a new one based (if there is no existing one).
    val spark = SparkSession.builder.appName("Reader")
      .config("spark.master", "local").getOrCreate
// Read the csv file
    val dict = spark.read
      .option("delimiter", "|")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath + "/dictionary.txt")

    val labels = spark.read
      .option("delimiter", "|")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath + "/sentiment_labels.txt")

    dict.join(labels, usingColumn = "ItemID")
  }
}
