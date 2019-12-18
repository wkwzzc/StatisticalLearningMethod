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
case class KmeansModel(data: DataFrame,
                       k: Int,
                       maxIter: Int = 40,
                       tol: Double = 1e-4) {

  private val spark: SparkSession = data.sparkSession
  private val weights: Array[Double] = new Array[Double](k).map(_ => 1.0 / k)
  private val ftsName: String = Identifiable.randomUID("KmeansModel2")

  @BeanProperty var fts: Array[String] = data.columns

  import spark.implicits._

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

  private val trainDf = dataTransForm(data)

  // step1 :随机选取 k个初始聚类中心
  var initk: Array[(Vector, Int)] = trainDf
    .randomSplit(weights, 1234)
    .map(df => df.head().getAs[Vector](ftsName))
    .zip(Range(0, k))

  implicit def vec2Seq(vec: Vector) = vec.toArray.toSeq

  /**
    *  判定赝本属于哪个类
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
    // step1 :随机选取 k个初始聚类中心

    // 结果表的Schema信息
    val schemaOfResult: StructType = data.schema
      .add("clusterId", IntegerType) //增加一列表示类id的字段

    // 创建一个空DF 用于接收结果
    var resultDF =
      spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schemaOfResult)

    while (i < maxIter && !convergence) {

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
      convergence = math.abs(newCost - cost) <= tol
      cost = newCost
      // 变换初始聚类中心
      initk = newItrs.map(_._1)
      i += 1 // 累加迭代次数
      resultDF = clustered.drop("cost", ftsName)

    }
    resultDF
  }

}
