package CH6_LogisticsRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import breeze.linalg.{DenseVector => densevector}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer.{mean => summaryMean}

/**
  * Created by WZZC on 2019/12/9
  **/
object lrRunner {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val iris = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("F:\\DataSource\\iris2.csv")

    val model =   LRModel(iris,"class")

    model.predict(iris).show(100)

    spark.stop()

  }
}
