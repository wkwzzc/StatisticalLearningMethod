package CH2_Perceptron
import breeze.linalg.{DenseVector => densevector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import scala.util.Random


/**
  * Created by WZZC on 2019/3/6
  * 通用感知机模型
  **/
object pocketPla {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val data = spark.read
      .option("inferSchema", true)
      .option("header", true)
      .csv("F:\\DataSource\\pocketPla.csv")

    val schema = data.schema
    val fts = schema.filterNot(_.name == "lable").map(_.name).toArray

    val amountVectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(fts)
      .setOutputCol("features")

    val vec2Array = udf((vec: DenseVector) => vec.toArray)

    val dataFeatrus = amountVectorAssembler
      .transform(data)
      .select($"lable", vec2Array($"features") as "features")
      .cache()

    var initW: densevector[Double] = densevector.rand[Double](fts.length) //创建一个初始化的随机向量
    var initb: Double = Random.nextDouble()
    var flag = true
    val lrate = 0.1 // 学习率
    var iteration = 0 //迭代次数

    var countError = dataFeatrus.count() //初始化错判个数（取样本大小）
    var resW = initW
    var resB = initb

    // 定义判别函数
    val signudf = udf((t: Seq[Double], y: Double) => {
      val wx = initW.dot(densevector(t.toArray))
      val d = wx + initb
      val ny = if (d >= 0) 1 else -1
      ny
    })

    while (flag  && iteration < 200) {

      val df = dataFeatrus.withColumn("sign", signudf($"features", $"lable"))
      val loss = df.where($"sign" =!= $"lable")
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

      println(s"迭代第${iteration}次 error:" + count)

      if (count == 0) {
        flag = false
      } else {
        // w1 = w0 + ny1x1
        //随机选择一个误判样本
        val rand = Random.nextInt(loss.count().toInt) + 1

        val randy = loss
          .withColumn("r", row_number().over(Window.orderBy($"lable")))
          .where($"r" === rand)
          .head()

        val y = randy.getAs[Int]("lable")
        initW = initW + densevector(
          randy.getAs[Seq[Double]]("features").toArray
        ).map(_ * y * lrate)
        // b1 = b0 + y
        initb = initb + y * lrate

      }
      iteration += 1

    }

    println(countError, resW, resB)

    // 定义判别函数
    val signudfres = udf((t: Seq[Double], y: Double) => {
      val wx = resW.dot(densevector(t.toArray))
      val d = wx + resB
      val ny = if (d >= 0) 1 else -1
      ny
    })

    val df = dataFeatrus.withColumn("sign", signudfres($"features", $"lable"))

    df.show(100)

    spark.stop()
  }
}
