package CH4_NaiveBayes

import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.Summarizer.{
  mean => summaryMean,
  variance => summaryVar
}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType

/**
  * Created by WZZC on 2019/4/27
  **/
object NaiveBayesGauss {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // 数据加载
    val irisData = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("F:\\DataSource\\iris.csv")

    val rolluped = irisData.rollup($"class").count()

    rolluped.show(100)

    // 样本量
    val sampleSize = rolluped.where($"class".isNull).head().getAs[Long](1)

    // 计算先验概率
    val pprobMap: Map[String, Double] = rolluped
      .where($"class".isNotNull)
      .withColumn("pprob", $"count" / sampleSize)
      .collect()
      .map(row => {
        row.getAs[String]("class") -> row.getAs[Double]("pprob")
      })
      .toMap

    val schema: StructType = irisData.schema
    val fts = schema.filterNot(_.name == """class""").map(_.name).toArray

    // 数据转换
    val amountVectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol("features")

    val ftsDF = amountVectorAssembler
      .transform(irisData)
      .select("class", "features")

    // 聚合计算：计算特征的均值向量和方差向量
    val irisAggred = ftsDF
      .groupBy($"class")
      .agg(
        summaryMean($"features") as "mfts",
        summaryVar($"features") as "vfts"
      )


    val cprobs: Array[(Array[(Double, Double)], String)] = irisAggred
      .collect()
      .map(row => {
        val cl = row.getAs[String]("class")
        val mus: Array[Double] = row.getAs[DenseVector]("mfts").toArray//
        val vars: Array[Double] = row.getAs[DenseVector]("vfts").toArray//
        (mus.zip(vars), cl)
      })

    cprobs.foreach(x=>{
      val str = x._1.mkString(",")

      val a = x._2
      println(a,str)
    })


    def pdf(x: Double, mu: Double, sigma2: Double) = {
      Gaussian(mu, math.sqrt(sigma2)).pdf(x)
    }

    val predictUDF = udf((vec: DenseVector) => {
      cprobs
        .map(tp => {
          val tuples: Array[((Double, Double), Double)] = tp._1.zip(vec.toArray)
          val cp: Double = tuples.map {
            case ((mu, sigma), x) => pdf(x, mu, sigma)
          }.product
          val pprob: Double = pprobMap.getOrElse(tp._2, 0)
          (cp * pprob, tp._2)
        })
        .maxBy(_._1)
        ._2
    })

    val predictDF = ftsDF
      .withColumn("predict", predictUDF($"features"))
    //    .select("class", "predict")

    predictDF.show()
//    predictDF.where($"class" =!= $"predict").show(truncate = false)

    spark.stop()
  }

}
