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
      .format("csv")
      .option("inferSchema", true)
      .option("header", true)
       .load("/Users/didi/IdeaProjects/StatisticalLearningMethod/src/main/resources/data/pla.csv")

    val perceptron = PerceptronModel(data)

    perceptron.setLabel("label")
    perceptron.setLrate(0.2)
    perceptron.fit


    perceptron.predict(data).show()

    spark.stop()
  }
}
