package CH9_EM
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by WZZC on 2020/5/30
  **/
object GMMModelRunner {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.shuffle.consolidateFiles", "true")
      .config("spark.io.compression.codec", "snappy")
      .getOrCreate()

    val dataset: DataFrame = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("data/gmm.txt")

    val fts = Array("f1", "f2", "f3")

    val model: GMMModel = new GMMModel(dataset)
    model.setFts(fts)
    model.setK(2)
    model.setMaxIter(5)

    model.fit

    val alphas = model.alpha
    val mus = model.mu
    val covs = model.cov

    println("================")
    alphas.foreach(a => println("alpha: " + a))
    println("================")
    covs.foreach(println)
    println("================")
    mus.foreach(println)

    model.predict.show(false)
    spark.stop()

  }

}
