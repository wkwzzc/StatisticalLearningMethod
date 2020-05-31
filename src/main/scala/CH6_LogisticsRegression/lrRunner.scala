package CH6_LogisticsRegression

import org.apache.spark.sql.SparkSession

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

    val iris = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("F:\\DataSource\\iris2.csv")

    val model: LRModel = LRModel(iris)

    model.setLabelColName("class")
    model.setFts(iris.columns.filterNot(_ == "class"))
    model.fit

    model.predict(iris).show(100)

    spark.stop()

  }
}
