package CH4_NaiveBayes
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.DoubleType

/**
  * Created by WZZC on 2019/4/27
  **/
object NaiveBayesRunner {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    // 数据加载
    val data = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("/Users/didi/IdeaProjects/StatisticalLearningMethod/src/main/resources/data/naviebayes.csv")

    val model = new StringIndexer()
      .setInputCol("x2")
      .setOutputCol("indexX2")
      .fit(data)

    val dataFrame: DataFrame = model
      .transform(data)
      .withColumn("x1", $"x1".cast(DoubleType))
      .withColumn("y", $"y".cast(DoubleType))

    val bayes: MultinomilNaiveBayesModel = MultinomilNaiveBayesModel(dataFrame)

    bayes.setLabelColName("y")
    bayes.setFts(Array("x1", "indexX2"))
    bayes.fit

    bayes.predict(dataFrame).show()

    spark.stop()
  }

}
