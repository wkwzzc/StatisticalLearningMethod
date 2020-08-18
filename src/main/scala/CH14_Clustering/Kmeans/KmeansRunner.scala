package CH14_Clustering.Kmeans






import org.apache.spark.sql.{DataFrame, SparkSession}





/**
 * Created by WZZC on 2019/12/18
 **/

object KmeansRunner  extends App {



  val spark = SparkSession
    .builder()
    .appName(s"${this.getClass.getSimpleName}")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  val iris: DataFrame = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("/Users/didi/IdeaProjects/StatisticalLearningMethod/src/main/resources/data/iris2label.csv")
    .drop("class")


    val model: KmeansModel = KmeansModel(iris)

    model.setFts(iris.columns)
    model.setK(2)


  model.fit
    .orderBy($"clusterId")
    .show(150)



  spark.stop()





}