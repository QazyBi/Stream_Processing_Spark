import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql
import org.apache.spark.sql.functions._

// class used to make the data ready for applying the ML models
object Preprocess {
 
  // filter out unnecessary segments 
  def filter(data: sql.DataFrame): sql.DataFrame = {
    
    // removing punctuation marks
    val remove_tokens = data.withColumn("SentimentArray", regexp_replace(data("SentimentText"), "['.,()!`]", ""))
    val separate = remove_tokens.withColumn("SentimentArray", functions.split(functions.trim(functions.col("SentimentArray")), "\\W+"))
    
    // remove common stop words
    val remove_stop = new StopWordsRemover()
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
      .setInputCol("SentimentArray")
      .setOutputCol("FilteredSentimentText")
      .setCaseSensitive(false)
    
    // apply transformation and return the resulting dataframe
    var output: DataFrame = remove_stop.transform(separate)
    output
  }
}
