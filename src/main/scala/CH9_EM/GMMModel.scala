package CH9_EM

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector}

import scala.beans.BeanProperty
import breeze.linalg.{
  *,
  diag,
  DenseMatrix => denseMatrix,
  DenseVector => densevector
}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StructType}

/**
 * Created by WZZC on 2020/5/17
 **/
case class GMMModel(data: DataFrame) {

  @BeanProperty var maxIter: Int = 40
  @BeanProperty var tol: Double = 1e-5
  @BeanProperty var k: Int = _
  @BeanProperty var fts: Array[String] = data.columns

  //  private val spark: SparkSession = data.sparkSession

  private val numFeatures = fts.length
  private val dataSize = data.count()

  // 初始化 alpha、 mu和sigma向量

  var alpha: Array[Double] = _

  var mu: Array[densevector[Double]] = _

  var cov: Array[denseMatrix[Double]] = _

  private def matrixArr: densevector[Double] =
    densevector(densevector.rand(numFeatures).toArray)
      .asInstanceOf[densevector[Double]]

  // 结果表的Schema信息
  val schemaOfResult: StructType = data.schema
    .add("clusterId", IntegerType) //增加一列表示类id的字段

  private var clusterDf: DataFrame = _
  //  var clusterDf  = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schemaOfResult)

  private val ftsName: String = Identifiable.randomUID("GMMModel")

  /**
   * 数据特征转换
   *
   * @param dataFrame
   * @return
   */
  def dataTransForm(dataFrame: DataFrame) = {
    new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol(ftsName)
      .transform(dataFrame)
  }

  /**
   * 计算多元高斯分布的概率密度函数
   *
   * @param vec  待计算样本
   * @param mus  均值向量
   * @param covs 协方差矩阵
   * @return
   */
  private def pdf(vec: densevector[Double],
                  mus: densevector[Double],
                  covs: denseMatrix[Double]) = {

    val muvec: DenseVector = new DenseVector(mus.data)
    val yvec: DenseVector = new DenseVector(vec.data)
    val covMatrix =
      new DenseMatrix(covs.rows, covs.cols, covs.data, covs.isTranspose)

    val mgaussian: MultivariateGaussian =
      new MultivariateGaussian(muvec, covMatrix)
    //    val mgaussian: MultivariateGaussian = MultivariateGaussian(mus, covs)
    //    使用breeze包的多元高斯分布会出现以下错误
    //    breeze.linalg.NotConvergedException
    mgaussian.pdf(yvec)

  }

  /**
   *
   * @return
   */
  private def gamakudf(mus: Array[densevector[Double]],
                       covs: Array[denseMatrix[Double]],
                       alphas: Array[Double]) =
    udf((vec: Vector) => {

      val tuples: Array[((densevector[Double], denseMatrix[Double]), Double)] =
        mus.zip(covs).zip(alphas)

      val featureVec = new densevector(vec.toArray)

      val gammak: Array[Double] = tuples.map(tp => {
        pdf(featureVec, tp._1._1, tp._1._2) * tp._2
      })

      val s = gammak.sum

      //      gammak.map(_/s)

      new DenseVector(gammak.map(x => (x / s).formatted("%.4f").toDouble))
      //      new DenseVector(gammak)

    })



  private def pdfudf(mus: Array[densevector[Double]],
                     covs: Array[denseMatrix[Double]]) =
    udf((vec: Vector) => {

      val tuples = mus.zip(covs)

      val featureVec = new densevector(vec.toArray)

      val gammak: Array[Double] = tuples.map(tp => {
        pdf(featureVec, tp._1, tp._2)
      })

      new DenseVector(gammak)

    })

  /**
   *
   * @param matrix
   * @param side
   */
  private def matrixToVectors(matrix: denseMatrix[Double], side: String) = {

    val rows: Int = matrix.rows
    val cols: Int = matrix.cols

    side match {
      case "row" => {
        (0 until rows).toArray
          .map(i => {
            densevector((0 until cols).toArray.map(j => {
              matrix.valueAt(i, j)
            }))
          })
      }

      case "col" => {
        (0 until cols).toArray
          .map(i => {
            densevector((0 until rows).toArray.map(j => {
              matrix.valueAt(j, i)
            }))
          })
      }
    }

  }

  def fit = {

    var alphas: Array[Double] = new Array[Double](k).map(_ => 1.0 / k)

    var mus: Array[densevector[Double]] =
      Array.fill(k)(densevector.rand(numFeatures))

    var covs: Array[denseMatrix[Double]] =
      Array.fill(k)(diag(matrixArr))

    var i = 0

    // TODO 添加代价函数的计算
    //      var cost = 0.0
    //      var convergence = false //判断收敛，即代价函数变化小于阈值tol

    while (i < maxIter) {

      //  E步:根据当前模型参数，计算分模型k对观测数据yi的响应度
      val edf: DataFrame = dataTransForm(data)
        .withColumn("gammajk", gamakudf(mus, covs, alphas)(col(ftsName)))

      val gammajk: RDD[densevector[Double]] = edf
        .select("gammajk")
        .rdd
        .map(x => densevector(x.getAs[Vector](0).toArray))

      //  M步:计算新一轮迭代的模型参数
      val gammaMatrix: denseMatrix[Double] = gammajk
        .map(_.toDenseMatrix)
        .reduce((v1, v2) => denseMatrix.vertcat(v1, v2))
        .t

      val sumgammak: Array[Double] = gammajk
        .map(_.toArray)
        .reduce((a1, a2) => a1.zip(a2).map(x => x._1 + x._2))

      val dataRdd = dataTransForm(data)
        .select(ftsName)
        .rdd
        .persist()

      val gammaVectors: Array[densevector[Double]] =
        matrixToVectors(gammaMatrix, "row")

      val dfmatrix: denseMatrix[Double] = dataRdd
        .map(x => densevector(x.getAs[Vector](0).toArray))
        .map(_.toDenseMatrix)
        .reduce((v1, v2) => denseMatrix.vertcat(v1, v2))

      //////////////////////////////////////
      // 更新sigmas
      covs = mus
        .zip(gammaVectors)
        .map(tp => {

          val ymusMatrix: denseMatrix[Double] = dataRdd
            .map(x => {
              val ym = densevector(x.getAs[Vector](0).toArray)
              ym - tp._1
            })
            .map(_.toDenseMatrix)
            .reduce((v1, v2) => denseMatrix.vertcat(v1, v2))

          (ymusMatrix(::, *) *:* tp._2).t * ymusMatrix

        })
        .zip(sumgammak)
        .map(tp => {
          tp._1.map(x => (x / tp._2).formatted("%.4f").toDouble)
        })

      ////////////////////////////////////////
      // 更新 mus
      mus = matrixToVectors(gammaMatrix, "row")
        .map(vec => {
          (vec.toDenseMatrix * dfmatrix).toDenseVector
        })
        .zip(sumgammak)
        .map(tp => {
          tp._1.map(_ / tp._2)
        })

      // 更新 alphas
      alphas = sumgammak.map(_ / dataSize).map(_.formatted("%.4f").toDouble)

      i += 1
      if (i >= maxIter) clusterDf = edf
    }

    alpha = alphas
    mu = mus
    cov = covs

  }

  //TODO   predict单样本计算
  def predict(seq: Seq[Double]) = {}


  def predict = {

    val predictUdf = udf((probability: Vector) => {
      val array = probability.toArray
      array.indexOf(array.max)
    })

    clusterDf
      .withColumnRenamed("gammajk", "probability")
      .withColumn("prediction", predictUdf(col("probability")))
      .drop(ftsName)

  }

}

