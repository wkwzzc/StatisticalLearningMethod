package CH5_DecisionTree

import org.apache.spark.sql._

/**
  * Created by WZZC on 2019/8/23
  **/
object ID3Ruuner {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val df: Dataset[Row] = spark.read
      .option("header", true)
      .csv("data/ID3.csv")

    val model: DecisionTreeModel = DecisionTreeModel(df)

    model.setLabelColName("label")
    model.setThreshold(1e-2)
    model.fit

    model.predict(df).show()

    spark.stop()
  }

}
