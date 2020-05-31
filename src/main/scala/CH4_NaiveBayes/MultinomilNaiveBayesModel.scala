package CH4_NaiveBayes

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.beans.BeanProperty
import scala.collection.mutable

/**
  * Created by WZZC on 2019/12/10
  **/
case class MultinomilNaiveBayesModel(data: DataFrame ) {

  private val spark: SparkSession = data.sparkSession

  @BeanProperty var labelColName:String = _
  @BeanProperty var fts: Array[String] =
    data.columns.filterNot(_ == labelColName)
  private val ftsName: String = Identifiable.randomUID("NaiveBayesModel")

  // 拉普拉斯平滑指数
  @BeanProperty var lamada: Double = 1.0

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

  var contingentProb: Array[(Double, (Int, Seq[mutable.Map[Double, Double]]))] =
    _

  var priorProb: Map[Double, Double] = _

  def fit = {

    def seqOp(c: (Int, Seq[mutable.Map[Double, Int]]),
              v: (Int, Seq[Double])) = {

      val w: Int = c._1 + v._1

      val tuples = c._2
        .zip(v._2)
        .map(tp => {
          val ftsvalueNum: Int = tp._1.getOrElse(tp._2, 0)

          tp._1 += tp._2 -> (ftsvalueNum + 1)
          tp._1
        })

      (w, tuples)

    }

    def combOp(c1: (Int, Seq[mutable.Map[Double, Int]]),
               c2: (Int, Seq[mutable.Map[Double, Int]])) = {

      val w = c2._1 + c1._1

      val resMap = c1._2
        .zip(c2._2)
        .map(tp => {
          val m1: mutable.Map[Double, Int] = tp._1
          val m2: mutable.Map[Double, Int] = tp._2

          m1.foreach(kv => {
            val i = m2.getOrElse(kv._1, 0)
            m2 += kv._1 -> (kv._2 + i)
          })

          m2
        })

      (w, resMap)
    }

    val nilSeq = new Array[Any](fts.length)
      .map(x => mutable.Map[Double, Int]())
      .toSeq

    val agged = dataTransForm(data).rdd
      .map(row => {

        val lable: Double = row.getAs[Double](labelColName)
        val fts: Seq[Double] = row.getAs[Vector](ftsName).toArray.toSeq

        (lable, (1, fts))
      })
      .aggregateByKey[(Int, Seq[mutable.Map[Double, Int]])]((0, nilSeq))(
        seqOp,
        combOp
      )

    val numLabels: Long = agged.count()
    val numDocuments: Double = agged.map(_._2._1).sum

    // 拉普拉斯变换
    val lpsamples: Double = numDocuments + lamada * numLabels

    //  条件概率
    contingentProb = agged
      .mapValues(tp => {
        val freq = tp._1

        val lprob = tp._2.map(
          m =>
            m.map {
              case (k, v) => {
                val logprob = (v / freq.toDouble).formatted("%.4f")
//                  val logprob = math.log(v / freq.toDouble).formatted("%.4f")
                (k, logprob.toDouble)
              }
          }
        )
        (freq, lprob)
      })
      .collect()

    //  先验概率
    priorProb = agged
      .map(tp => (tp._1, tp._2._1))
      .mapValues(v => ((v + lamada) / lpsamples))
//      .mapValues(v => math.log((v + lamada) / lpsamples))
      .collect()
      .toMap

  }

  def predict(predictData: Seq[Double]): Double = {

    val posteriorProb: Array[(Double, Double)] = contingentProb
      .map(tp3 => {
        val label: Double = tp3._1

        val tp: (Int, Seq[mutable.Map[Double, Double]]) = tp3._2

        val missProb: Double = lamada / (tp._2.length * lamada)

        val sum: Double = tp._2
          .zip(predictData)
          .map {
            case (pmap, ftValue) => {

              val d: Double = pmap.getOrElse(ftValue, missProb)

              math.log(d)

            }
          }
          .sum

        (label, sum)

      })
      .map(tp => (tp._1, priorProb.getOrElse(tp._1, 0.0) + math.log(tp._2)))

    posteriorProb.maxBy(_._2)._1

  }

  def predict(predictData: DataFrame): DataFrame = {

    contingentProb.foreach(println)

    priorProb.foreach(println)

    val predictudf = udf((vec: Vector) => predict(vec.toArray.toSeq))

    dataTransForm(predictData)
      .withColumn(labelColName, predictudf(col(ftsName)))
      .drop(ftsName)

  }

}
