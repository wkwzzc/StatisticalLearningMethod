package CH4_NaiveBayes

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by WZZC on 2019/3/13
  **/
object NaiveBayesMultinomil {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    // 数据加载
    val mnistTrain = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("F:\\DataSource\\data\\mnist\\train.csv")

    // Schema信息
    val ftSchemas = mnistTrain.schema.map(_.name).filterNot(_ == "label")

    // 拉普拉斯平滑指数
    val lamada = 1.0

    // 数据格式变换
    val flattenDF = mnistTrain
      .flatMap(row => {
        val label: Int = row.getInt(0)
        val tuples = (1 until row.length).map(i => {
          (label, ftSchemas(i - 1), row.getInt(i))
        })
        tuples
      })
      .toDF("label", "ftsName", "ftsValue")

    // 分组计算每个分类的各个特征出现的频次
    val grouped = flattenDF
      .groupBy($"label", $"ftsName", $"ftsValue")
      .agg(count($"ftsValue") as "ftsFreq")
      .persist()

    val ftsLevels = grouped
      .groupBy($"ftsName")
      .agg(countDistinct($"ftsValue") as "ftsLevels")

    val labelLevels = grouped
      .where($"ftsName" === ftSchemas.head)
      .groupBy($"label")
      .agg(sum($"ftsFreq") as "ftsCounts")

    grouped.unpersist()

    //分类数量
    val numLabels: Double = labelLevels.count().toDouble

    //样本量
    val numSample: Double = labelLevels.rdd
      .map(_.getLong(1).toDouble)
      .collect()
      .sum

    // 拉普拉斯变换
    val lpsamples: Double = numSample + lamada * numLabels

    // 计算先验概率和
    val pprobAndlp = labelLevels
      .crossJoin(ftsLevels)
      .withColumn("pprob", log(($"ftsCounts" + lamada) / lpsamples))
      .withColumn("lp", $"ftsCounts" + $"ftsLevels" * lamada)
      .drop("ftsCounts", "ftsLevels")

    // 取对数后的先验概率
    val pprobDF = pprobAndlp.select($"label", $"pprob").distinct()

    val ftsLevelLpsDF = pprobAndlp.select($"label", $"ftsName", $"lp")

    // 条件概率
    val cprobDF = grouped
      .join(ftsLevels, "ftsName")
      .join(labelLevels, "label")
      .select(
        $"label",
        $"ftsName",
        $"ftsValue",
        ($"ftsFreq" + lamada) / ($"ftsLevels" + $"ftsCounts") as "cprob"
      )

    val cprob: RDD[(String, (Map[Int, Double], Int))] = cprobDF.rdd
      .map(row => {
        val label = row.getInt(0)
        val ftsName = row.getString(1)
        val ftsValue = row.getInt(2)
        val logcProb = row.getDouble(3)
        ((label, ftsName), Map(ftsValue -> logcProb))
      })
      .reduceByKey(_ ++ _)
      .map {
        case ((label, features), cprobs) =>
          (features, (cprobs, label))
      }

    // ########################################预测####################################### //
    // 将条件概率的数据广播
    val cprobBroad = sc.broadcast(cprob.collect())

    // 先验概率
    val priorProbability = pprobDF
      .map(row => {
        row.getAs[Int]("label") -> row.getAs[Double]("pprob")
      })
      .collect()
      .toMap

    // ftsLevelLp
    val ftsLevelLps: Map[(Int, String), Double] = ftsLevelLpsDF
      .map(row => {
        val lb = row.getAs[Int]("label")
        val ftsName = row.getAs[String]("ftsName")
        val lp = row.getAs[Double]("lp")
        (lb, ftsName) -> lp
      })
      .collect()
      .toMap

    val predict = mnistTrain.rdd.map(row => {
      //  (label, prob)
      val labelAndProb: (Int, Double) = (1 until row.length)
        .map(i => {
          val observations = row.getInt(i) //第i个特征的观测值

          val cpCompute: Array[(Int, Double)] = cprobBroad.value
            .filter(_._1 == ftSchemas(i - 1))
            .map(tps => {
              // 特征i的条件概率Map
              val cpMap: Map[Int, Double] = tps._2._1
              // 拉普拉斯平滑防止出现条件概率为0
              val missFtscProb: Double = lamada / ftsLevelLps
                .getOrElse((tps._2._2, tps._1), lamada)
              //观测值的条件概率
              val maybeDouble = cpMap.get(observations)
              val cp: Double = maybeDouble match {
                case None => missFtscProb
                case _    => maybeDouble.head
              }
              (tps._2._2, math.log(cp))
            })

          cpCompute
        })
        .flatMap(_.toSeq)
        .groupBy(_._1)
        .mapValues(_.map(_._2).sum)
        .map(tp => {
          tp._1 -> (priorProbability.getOrElse(tp._1, 0.0) + tp._2)
        })
        .maxBy(_._2)

      Row.merge(row, Row.fromTuple(labelAndProb))
    })

    val newSchema = mnistTrain.schema
      .add("predict", IntegerType)
      .add("lprob", DoubleType)

    val predictdf = spark
      .createDataFrame(predict, newSchema)
      .withColumn("label", $"label".cast(DoubleType))
      .withColumn("predict", $"predict".cast(DoubleType))
      .cache()

    predictdf
      .select($"label", $"predict", $"lprob")
      .show(truncate = false)

    // 正确率
    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predict")
      .setMetricName("accuracy")
    val accuracy = evaluator1.evaluate(predictdf)
    println("正确率 =" + accuracy)

    // 召回率
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predict")
      .setMetricName("accuracy")
    val Recall = evaluator2.evaluate(predictdf)
    println("召回率 = " + Recall)

    spark.stop()

  }
}
