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
      .option("inferSchema", true)
      .option("header", true)
      .csv("/Users/didi/IdeaProjects/StatisticalLearningMethod/src/main/resources/data/iris.csv")

    val model = KnnModel(iris ,"class")


    //    println(model.fts.mkString("~"))

    model.predict(iris, 3).show(150)


    spark.stop()

  }

}
