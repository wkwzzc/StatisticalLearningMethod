package CH3_KNearestNeibor

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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
      .option("inferSchema", true)
      .option("header", true)
      .csv("F:\\DataSource\\iris.csv")
//      .repartition(2)

    val model =   KnnModel(iris, "class")

    val frame  = model.kdtrees

    frame .foreach(println)

    model.predict(iris,3).show(100)

//    frame.foreach(x=>println(x.value))

    spark.stop()

  }

}
