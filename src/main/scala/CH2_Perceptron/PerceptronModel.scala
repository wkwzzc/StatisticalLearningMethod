package CH2_Perceptron

import java.util.UUID

import breeze.linalg.{DenseVector => densevector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number, udf}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import statisticslearn.DataUtils.udfFuns

import scala.util.Random

/**
  * Created by WZZC on 2019/11/28
  **/
case class PerceptronModel(data: DataFrame, label: String, lrate: Double)
    extends Serializable {

  private val spark: SparkSession = data.sparkSession
  private val sc = spark.sparkContext
  import spark.implicits._

  private val labelCol: Column = new Column(label) with Serializable

  private val featuresName: String = UUID.randomUUID().toString
  private val featuresCol = new Column(featuresName) with Serializable

  // 数据特征名称
  val fts: Array[String] = data.columns.filterNot(_ == label)

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
    val amountVectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol(featuresName)

    amountVectorAssembler
      .transform(dataFrame)
      .withColumn(featuresName, udfFuns.vec2Array(featuresCol))

  }

  /**
    * PLA 模型拟合
    *
    * @return
    */
  def fit = {

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
        ).map(_ * y * lrate)
        // b1 = b0 + y
        initb = initb + y * lrate
      }
    }

    (initW, initb)
  }

  /**
    *  PlA 口袋算法，可以针对线性不可分的数据集构建感知机模型，但是准确率会下降
    *
    * @param iter 迭代次数
    * @return
    */
  def pocketPlaFit(iter: Int) = {

    var initW: densevector[Double] = densevector.rand[Double](fts.length) //创建一个初始化的随机向量
    var initb: Double = Random.nextDouble()
    var flag = true

    val dataFeatrus: DataFrame = dataTransForm(this.data)
      .select(labelCol, featuresCol)

    var iteration = 0 //迭代次数
    val allCount = dataFeatrus.count()
    var countError = allCount //初始化错判个数（取样本大小）
    var resW = initW
    var resB = initb

    while (flag && iteration < iter) {

      val df =
        dataFeatrus.withColumn("sign", signudf(initW, initb)(featuresCol))
      val loss = df.where($"sign" =!= labelCol)
      val count = loss.count().toInt

      /**
        * 判断新模型的误判次数是否小于前一次的误判次数
        * 如果小于则更新权值向量和偏置，大于则不更新
        * */
      if (count < countError) {
        countError = count
        resW = initW
        resB = initb
      }

      if (count == 0) {
        flag = false
      } else {
        // w1 = w0 + ny1x1
        //随机选择一个误判样本
        val rand = Random.nextInt(loss.count().toInt) + 1

        val randy = loss
          .withColumn("r", row_number().over(Window.orderBy(labelCol)))
          .where($"r" === rand)
          .head()

        val y = randy.getAs[Int](label)
        initW = initW + densevector(
          randy.getAs[Seq[Double]](featuresName).toArray
        ).map(_ * y * lrate)
        // b1 = b0 + y
        initb = initb + y * lrate

      }
      iteration += 1

    }

    (resW, resB, countError.toDouble / allCount)

  }

  /**
    *  PLA  预测
    *
    * @param predictDf
    * @param w
    * @param b
    * @return
    */
  def predict(predictDf: DataFrame, w: densevector[Double], b: Double) = {
    val transFormed: DataFrame = dataTransForm(predictDf)
    transFormed
      .withColumn(label, signudf(w, b)(featuresCol))
      .drop(featuresName)
  }

}
