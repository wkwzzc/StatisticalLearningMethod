package CH3_KNearestNeibor

import org.apache.spark.sql.SparkSession

/**
  * Created by WZZC on 2019/11/29
  **/
object KNNRunner {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    val iris = spark.read
//      .format("csv")
      .option("inferSchema", true)
      .option("header", true)
      .csv("F:\\DataSource\\iris.csv")

    val model = KnnModel(iris, "class")

    model.predict(iris, 3).show(100)

    spark.stop()

  }

}
