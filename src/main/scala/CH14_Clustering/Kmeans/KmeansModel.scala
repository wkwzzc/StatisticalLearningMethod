package CH14_Clustering.Kmeans

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Identifiable
import statisticslearn.DataUtils.distanceUtils.euclidean
import org.apache.spark.ml.stat.Summarizer.{mean => summaryMean}
import org.apache.spark.sql.types.{IntegerType, StructType}
import scala.beans.BeanProperty

/**
 * Created by WZZC on 2019/12/17
 **/
case class KmeansModel(data: DataFrame) {

  @BeanProperty var maxIter: Int = 40
  @BeanProperty var tol: Double = 1e-8
  @BeanProperty var k: Int = _
  @BeanProperty var fts: Array[String] = data.columns


  // TODO 自定义指定初始聚类中心，可以提高聚类的准确性以及运行性能

  private val spark: SparkSession = data.sparkSession

//  def  weights: Array[Double] = new Array[Double](this.getK).map(_ => 1.0 / this.getK)
  private val ftsName: String = Identifiable.randomUID("KmeansModel2")

  import spark.implicits._

  /**
   * 数据特征转换
   *
   * @param dataFrame
   * @return
   */
  def dataTransForm(dataFrame: DataFrame) = {
    new VectorAssembler()
      .setInputCols(this.getFts)
      .setOutputCol(ftsName)
      .transform(dataFrame.select(   ))
  }

  private def trainDf = dataTransForm(data)



  implicit def vec2Seq(vec: Vector) = vec.toArray.toSeq

  /**
   * 判定样本属于哪个类
   *
   * @param center 聚类中心
   * @return
   */
  def cluserUdf(center: Array[(Vector, Int)]) =
    udf((fts: Vector) => {
      center
        .map {
          case (vector, clusterId) =>
            val d: Double = euclidean(vector, fts)
            (clusterId, d)
        }
        .minBy(_._2)
    })

  def fit: DataFrame = {

    var i = 0 // 迭代次数
    var cost = 0.0 //初始的代价函数
    var convergence = false //判断收敛，即代价函数变化小于阈值tol


    // step1 :选取k个初始聚类中心（为了方便以及性能，选择前k个样本作为初始聚类中心）
    var initk: Array[(Vector, Int)] = trainDf.head(k)
      .map(row => row .getAs[Vector](ftsName))
      .zip(Range(0, this.getK))

    // 结果表的Schema信息
    val schemaOfResult: StructType = data.schema
      .add("clusterId", IntegerType) //增加一列表示类id的字段

    // 创建一个空DF 用于接收结果
    var resultDF =
      spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schemaOfResult)

    while (i < this.getMaxIter && !convergence) {

      val clustered = trainDf
        .withColumn("clust", cluserUdf(initk)(col(ftsName)))
        .withColumn("clusterId", $"clust".getField("_1"))
        .withColumn("cost", $"clust".getField("_2"))
        .drop("clust")

      val newItrs: Array[((Vector, Int), Double)] = clustered
        .groupBy($"clusterId")
        .agg(sum($"cost") as "cost", summaryMean(col(ftsName)) as ftsName)
        .collect()
        .map(row => {
          val clusterId = row.getAs[Int]("clusterId")
          val cost = row.getAs[Double]("cost")
          val ftsValues: Vector = row.getAs[Vector](ftsName)
          ((ftsValues, clusterId), cost)
        })

      val newCost: Double = newItrs.map(_._2).sum
      println(s"第${i} 次迭代 ；新的损失函数为 ${newCost}")

      convergence = math.abs(newCost - cost) <= this.getTol
      cost = newCost
      // 变换初始聚类中心
      initk = newItrs.map(_._1)
      i += 1 // 累加迭代次数
      resultDF = clustered.drop("cost", ftsName)

    }
    resultDF
  }

}
