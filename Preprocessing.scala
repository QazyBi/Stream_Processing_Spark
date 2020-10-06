import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions.col
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions}

object Preprocessing {
  def filterData(data: sql.DataFrame): sql.DataFrame = {
// Remove uneeded characters
    val remove_signs = data.withColumn("SentimentArray",
      regexp_replace(data("SentimentText"), "['.,()!`]", ""))
    val separated_data = remove_signs.withColumn("SentimentArray",
      functions.split(functions.trim(functions.col("SentimentArray")), "\\W+"))
// Filter our data
    val filter = new StopWordsRemover()
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
      .setInputCol("SentimentArray")
      .setOutputCol("FilteredSentimentText")
      .setCaseSensitive(false)

    var filtered: DataFrame = filter.transform(separated_data)
    filtered = filtered.filter(size(col("FilteredSentimentText")) >= 1)

    filtered
  }

  def preprocess(data: sql.DataFrame): sql.DataFrame = {
    filterData(data)
  }
}
