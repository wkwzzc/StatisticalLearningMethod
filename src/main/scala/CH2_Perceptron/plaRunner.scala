package CH2_Perceptron

import breeze.linalg.DenseVector
import org.apache.spark.sql.SparkSession

/**
  * Created by WZZC on 2019/3/4
  *
  **/
object plaRunner {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    val data = spark.read
      .option("inferSchema", true)
      .option("header", true)
      .csv("F:\\DataSource\\pla.csv")

    val perceptron = PerceptronModel(data, "lable", 0.2)

    val fit: (DenseVector[Double], Double) = perceptron.plaFit
    perceptron.predict(data, fit._1, fit._2).show(data.count().toInt)

    val fit2: (DenseVector[Double], Double, Double) =
      perceptron.pocketPlaFit(100)
    perceptron.predict(data, fit2._1, fit2._2).show(data.count().toInt)

    spark.stop()
  }
}