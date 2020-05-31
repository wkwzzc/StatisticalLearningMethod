package CH14_Clustering.Kmeans
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by WZZC on 2019/12/18
  **/
object KmeansRunner {
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
      .drop("class")

    val model = KmeansModel(iris )
    model.setK(2)
    model.setFts(iris.columns.filterNot(_ == "class"))

    val res: DataFrame = model.fit
    res.show(100, false)

    spark.stop()

  }

}
