package CH4_NaiveBayes

import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.Summarizer.{
  mean => summaryMean,
  variance => summaryVar
}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf

/**
  * Created by WZZC on 2019/12/10
  **/
case class NaiveBayesModel(data: DataFrame, labelColName: String) {

  private val spark: SparkSession = data.sparkSession
  import spark.implicits._

  private val labelColumn: Column = col(labelColName)
  private val fts: Array[String] = data.columns.filterNot(_ != labelColName)

  /**
    * 数据特征转换
    *
    * @param dataFrame
    * @return
    */
  def dataTransForm(dataFrame: DataFrame) = {
    new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol("features")
      .transform(dataFrame)
  }

  /**
    *  计算先验概率
    *
    * @return
    */
  def priorProb = {
    val rolluped: DataFrame = data.rollup(labelColName).count()

    // 样本量
    val sampleSize: Long =
      rolluped.where(labelColumn.isNull).head.getAs[Long](1)

    // 计算先验概率
    val priorProbMap: Map[String, Double] = rolluped
      .where(labelColumn.isNotNull)
      .withColumn("pprob", $"count" / sampleSize)
      .collect()
      .map(row => {
        row.getAs[String](labelColName) -> row.getAs[Double]("pprob")
      })
      .toMap

    priorProbMap

  }

  /**
    *  计算条件概率
    *
    * @return
    */
  def condProb = {
    dataTransForm(data)
      .groupBy(labelColumn)
      // 聚合计算：计算特征的均值向量和方差向量
      .agg(
        summaryMean($"features") as "mfts",
        summaryVar($"features") as "vfts"
      )
      .collect()
      .map(row => {
        val cl = row.getAs[String](labelColName)
        val mus = row.getAs[DenseVector]("mfts").toArray
        val vars = row.getAs[DenseVector]("vfts").toArray
        (mus.zip(vars), cl)
      })
  }

  /**
    *计算样本x的概率密度函数（正态分布）
    *
    * @param x  样本x
    * @param mu 正态分布的均值
    * @param sigma2 正态分布的方差
    * @return
    */
  def pdf(x: Double, mu: Double, sigma2: Double) = {
    Gaussian(mu, math.sqrt(sigma2)).pdf(x)
  }

  // 预测UDF
  private val predictUDF = udf((vec: DenseVector) => {
    condProb
      .map(tp => {
        val tuples = tp._1.zip(vec.toArray)
        val cp: Double = tuples.map {
          case ((mu, sigma), x) => pdf(x, mu, sigma)
        }.product
        val pprob: Double = priorProb.getOrElse(tp._2, 0)
        (cp * pprob, tp._2)
      })
      .maxBy(_._1)
      ._2
  })

  def predict(df: DataFrame) = {}
}
