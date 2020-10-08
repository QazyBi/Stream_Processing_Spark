import org.apache.spark.sql.{DataFrame, SparkSession}

// class used to read the training dataset
object Reader {

  def read_dataset(path: String): DataFrame = {
    // get an existing SparkSession or create a new one
    val spark = SparkSession.builder.appName("Reader")
      .config("spark.master", "local").getOrCreate
    
    // setting up the dictionary
    val dictionary = spark.read
      .option("delimiter", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(path + "/dictionary.txt")
    
    // setting up the dictionary
    val labels = spark.read
      .option("delimiter", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(path + "/sentiment_labels.txt")
    
    // joining read data on the index
    dictionary.join(labels, usingColumn = "ItemID")
  }
}
