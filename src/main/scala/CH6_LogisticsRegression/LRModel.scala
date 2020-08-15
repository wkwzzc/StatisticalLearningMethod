package CH6_LogisticsRegression

import org.apache.spark.sql.functions.{col, mean, udf}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  StringIndexerModel,
  VectorAssembler
}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseVector => densevector}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer.{mean => summaryMean}
import org.apache.spark.ml.util.Identifiable

import scala.beans.BeanProperty

/**
 * Created by WZZC on 2019/12/9
 **/
case class LRModel(data: DataFrame) {

  private val spark: SparkSession = data.sparkSession

  import spark.implicits._

  @BeanProperty var itr: Int = 40 //迭代次数
  @BeanProperty var lrate: Double = 0.05 //学习率
  @BeanProperty var error: Double = 1e-3 // 初始化差值
  @BeanProperty var fts: Array[String] = _
  @BeanProperty var labelColName: String = _

  private var w: densevector[Double] = _

  private val ftsName: String = Identifiable.randomUID("LRModel")
  private val indexedLabel: String = Identifiable.randomUID("indexedLabel")

  private def stringIndexer: StringIndexerModel = new StringIndexer()
    .setInputCol(this.getLabelColName)
    .setOutputCol(indexedLabel)
    .fit(data)

  def dataTransForm(df: DataFrame) = {
    new VectorAssembler()
      .setInputCols(this.getFts)
      .setOutputCol(ftsName)
      .transform(data)
  }

  // sigmoid function
  def sigmoid(x: Double) = 1 / (1 + math.exp(-x))

  def sigmoidUdf(initW: densevector[Double]) =
    udf((ftsVal: Vector) => {
      val d = initW.dot(densevector(ftsVal.toArray))
      sigmoid(d)
    })

  // 计算损失函数
  def lossUdf =
    udf((sigmoid: Double, y: Double) => y * sigmoid + (1 - y) * (1 - sigmoid))

  // 计算梯度下降
  def gradientDescentUdf =
    udf((ftsVal: Vector, y: Double, sigmoid: Double) => {
      val gd: Array[Double] = ftsVal.toArray.map(_ * (sigmoid - y))
      Vectors.dense(gd)
    })

  // 预测
  def predictUdf(w: densevector[Double]) =
    udf((ftsVal: Vector) => {
      val d: Double = w.dot(densevector(ftsVal.toArray))
      if (d >= 0) 1.0 else 0.0
    })

  private def fitModel = {
    var currentLoss: Double = Double.MaxValue //当前损失函数最小值
    var change: Double = error + 0.1 // 梯度下降前后的损失函数的差值
    var i = 0 // 迭代次数
    var initW: densevector[Double] = densevector.rand[Double](this.getFts.length)

    while (change > error & i < itr) {
      //创建一个初始化的随机向量作为初始权值向量

      val vecDf: DataFrame = dataTransForm(this.data)
      val sigmoidDf = stringIndexer
        .transform(vecDf)
        .select(ftsName, indexedLabel)
        .withColumn("sigmoid", sigmoidUdf(initW)(col(ftsName)))
        .cache()

      val loss = sigmoidDf
        .select(lossUdf($"sigmoid", col(indexedLabel)) as "loss")
        .agg(mean($"loss"))
        .head
        .getDouble(0)

      change = math.abs(currentLoss - loss)
      currentLoss = loss

      val gdVector: Vector = sigmoidDf
        .select(
          gradientDescentUdf(col(ftsName), col(indexedLabel), $"sigmoid") as "gd"
        )
        .agg(summaryMean($"gd") as "gd")
        .head
        .getAs[Vector]("gd")

      initW -= densevector(gdVector.toArray.map(_ * lrate))

      sigmoidDf.unpersist()
      i += 1
    }

    (initW, currentLoss)
  }

  def fit = {
    w = fitModel._1
  }

  def predict(df: DataFrame): DataFrame = {
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(stringIndexer.labels)

    val vecDf: DataFrame = dataTransForm(df)

    val preDf = vecDf.withColumn("prediction", predictUdf(w)(col(ftsName)))

    labelConverter
      .transform(preDf)
      .drop(ftsName, "prediction")
  }

}
