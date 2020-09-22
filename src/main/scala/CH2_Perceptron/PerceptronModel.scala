package CH2_Perceptron

import java.util.UUID

import breeze.linalg.{DenseVector => densevector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number, udf}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import statisticslearn.DataUtils.udfFuns

import scala.beans.BeanProperty
import scala.util.Random

/**
 * Created by WZZC on 2019/11/28
 **/
case class PerceptronModel(data: DataFrame)
  extends Serializable {

  private val spark: SparkSession = data.sparkSession
  private val sc = spark.sparkContext

  import spark.implicits._

  @BeanProperty var lrate: Double = 0.2 //学习率

  @BeanProperty var label: String = _ // 分类指标

  @BeanProperty def fts: Array[String] = data.columns.filterNot(_ == this.getLabel) // 数据特征名称


  private var W: densevector[Double] = _
  private var b: Double = _


  def labelCol: Column = new Column(this.getLabel)

  private val featuresName: String = UUID.randomUUID().toString
  private val featuresCol = new Column(featuresName) with Serializable


  // 定义判定函数
  def signudf(w: densevector[Double], b: Double) =
    udf((t: Seq[Double]) => {
      val wx: Double = w.dot(densevector(t.toArray))
      val d: Double = wx + b
      val ny = if (d >= 0) 1 else -1
      ny
    })


  /**
   * 数据转换
   *
   * @param dataFrame
   * @return
   */
  def dataTransForm(dataFrame: DataFrame) = {
    new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol(this.featuresName)
      .transform(dataFrame)
      .withColumn(this.featuresName, udfFuns.vec2Array(this.featuresCol))
  }

  /**
   * PLA 模型拟合
   *
   * @return
   */
  private def fitModel = {

    val dataFeatrus: DataFrame = dataTransForm(this.data)
      .select(labelCol, featuresCol)

    //创建一个初始化的随机向量作为初始权值向量
    var initW: densevector[Double] = densevector.rand[Double](fts.length)
    // 初始偏置
    var initb: Double = Random.nextDouble()

    var flag = true

    var resDf = spark.createDataFrame(
      sc.emptyRDD[Row],
      dataFeatrus.schema.add("nG", IntegerType)
    )

    while (flag) {
      val df =
        dataFeatrus.withColumn("sign", signudf(initW, initb)(featuresCol))

      val loss = df.where($"sign" =!= labelCol)

      val count: Long = loss.count()

      if (count == 0) {
        resDf = df
        flag = false
      } else {
        // w1 = w0 + ny1x1
        //随机选择一个误判样本
        val rand = Random.nextInt(loss.count().toInt) + 1

        val randy = loss
          .withColumn(
            "r",
            row_number().over(Window.partitionBy(labelCol).orderBy(labelCol))
          )
          .where($"r" === rand)
          .head()

        val y = randy.getAs[Int](labelCol.toString())

        // 更新 w 和 b
        initW = initW + densevector(
          randy.getAs[Seq[Double]](featuresName).toArray
        ).map(_ * y * this.getLrate)
        // b1 = b0 + y
        initb = initb + y * lrate
      }
    }

    (initW, initb)
  }

  def fit = {
    W = fitModel._1
    b = fitModel._2
  }


  /**
   * PLA  预测
   *
   * @param predictDf
   * @return
   */
  def predict(predictDf: DataFrame) = {
    val transFormed: DataFrame = dataTransForm(predictDf)
    transFormed
      .withColumn(this.getLabel, signudf(W, b)(featuresCol))
      .drop(featuresName)
  }


}
