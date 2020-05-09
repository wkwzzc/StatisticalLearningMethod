package CH4_NaiveBayes

import org.apache.spark.sql.SparkSession


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


    // 数据加载
    val irisData = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("F:\\DataSource\\iris.csv")


    val model = NaiveBayesModel(irisData, "class")

    model.predict(irisData).show(200)

    spark.stop()
  }

}
