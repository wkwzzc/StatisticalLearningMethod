package CH5_DecisionTree

import org.apache.spark.sql.functions.{col, count, log2, sum}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.beans.BeanProperty

/**
  * Created by WZZC on 2019/12/6
  **/
case class DecisionTreeModel(data: DataFrame) {

  private val spark = data.sparkSession
  import spark.implicits._

  @BeanProperty var threshold: Double = 1e-2
  @BeanProperty var labelColName: String = _
  var node: DtreeNode = _
  var search: List[String] = _

  /**
    *  获取实例数最大的类Ck
    *
    * @param dataFrame
    * @return
    */
  def maxCountLabel(dataFrame: DataFrame) = {
    dataFrame
      .select(labelColName)
      .groupBy(labelColName)
      .agg(count(labelColName) as "ck")
      .collect()
      .map(row => (row.getString(0), row.getLong(1)))
      .maxBy(_._2)
      ._1
  }

  /**
    *最优特征选择（ID3）
    *
    * @param df dataframe
    * @param labelCol ck类S Colname
    * @param ftSchemas 特征集合
    * @return
    */
  def optimalFeatureSel(df: DataFrame,
                        labelCol: String,
                        ftSchemas: Array[String]) = {

    // 数据格式转换，行转列
    val ftsCount = df
      .flatMap(row => {
        val label = row.getAs[String](labelColName)
        (0 until row.length).map(i => {
          (label, ftSchemas(i), row.getString(i))
        })
      })
      .toDF("label", "ftsName", "ftsValue")
      .groupBy("label", "ftsName", "ftsValue")
      .agg(count("label") as "lcount")
      .repartition($"ftsName")
      .cache()

    //impiricalEntropy 经验熵
    val preProbdf = ftsCount
      .where($"ftsName" === labelCol)
      .cache()

    val dfcount: Double = preProbdf
      .agg(sum("lcount") as "lsum")
      .head()
      .getAs[Long]("lsum")
      .toDouble

    val impiricalEntropy: Double = preProbdf
      .withColumn("pi", $"lcount" / dfcount)
      .withColumn("hd", log2($"pi") * (-$"pi"))
      .agg(sum($"hd") as "hd")
      .collect()
      .head
      .getAs[Double]("hd")

    //经验条件熵
    val ftsValueCount: DataFrame = ftsCount
      .filter($"ftsName" =!= labelCol)
      .groupBy($"ftsName", $"ftsValue")
      .agg(sum("lcount") as "lsum")

    val cens: DataFrame = ftsCount
      .join(ftsValueCount, Seq("ftsName", "ftsValue"))
      .orderBy($"ftsName", $"label")
      .withColumn("cpi", $"lcount".cast(DoubleType) / $"lsum")
      .withColumn("pi", $"lsum" / dfcount)
      .withColumn("gda", -$"cpi" * log2($"cpi") * $"pi")
      .groupBy($"ftsName")
      .sum("gda")

    // 信息增益 ->最大信息增益
    val (ftsName, maxGda) = cens
      .withColumn("xxzy", -$"sum(gda)" + impiricalEntropy)
      .collect()
      .map(row => {
        val ftsName = row.getString(0)
        val maxgda = row.getDouble(2)
        (ftsName, maxgda)
      })
      .maxBy(_._2)

    val ftsLabels: Array[String] = ftsValueCount
      .where($"ftsName" === ftsName)
      .select($"ftsValue")
      .collect()
      .map(_.getString(0))

    ftsCount.unpersist()
    preProbdf.unpersist()

    (ftsName, maxGda, ftsLabels)

  }

  /**
    * 数据按照特征的值划分
    *
    * @param df dataframe
    * @param ftsName 划分的特征名称
    * @param ftsLabels
    * @return
    */
  def splitByFts(df: DataFrame, ftsName: String, ftsLabels: Array[String]) = {
    val column: Column = col(ftsName)
    ftsLabels.map(ftsvalue => {
      ftsvalue -> df.where(column === ftsvalue).drop(ftsName)
    })
  }

  def fit = {

    var searchList: List[String] = Nil

    /**
      *
      * @param data  DataFrame
      * @param fNodeName 当前结点的特征名称
      * @return
      */
    def creatTree(data: DataFrame, fNodeName: String = null): DtreeNode = {

      data.persist()

      val datalabels = data.select(labelColName).distinct()
      // 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
      val dataCount: Long = datalabels.count()
      if (dataCount == 1) {
        val ck: String = datalabels.head().getString(0)
        DtreeNode(fNodeName, ck, Nil)
      }

      // 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
      // 数据特征名称
      val ftSchemas: Array[String] = data.columns
      if (ftSchemas.isEmpty) {
        val ck: String = maxCountLabel(data)
        DtreeNode(fNodeName, ck, Nil)
      }

      // 3,计算信息增益；判断信息增益是否小于阈值 ,小于则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
      // 不小于则递归创建树
      val (ftsName, maxgda, ftsLabels) =
        optimalFeatureSel(data, labelColName, ftSchemas)

      searchList = ftsName +: searchList

      val dtreeNode = if (maxgda < threshold) {
        val ck: String = maxCountLabel(data)
        DtreeNode(fNodeName, ck, Nil)
      } else {
        val nodaDfs: Array[(String, DataFrame)] =
          splitByFts(data, ftsName, ftsLabels)
        val nodes: Seq[(String, DtreeNode)] = nodaDfs
          .map(tp => {
            val ftsValue: String = tp._1
            val splitedDf: DataFrame = tp._2
            data.unpersist()
            ftsValue -> creatTree(splitedDf, ftsName)
          })
          .toSeq

        DtreeNode(ftsName, "", nodes)

      }

      dtreeNode

    }

    node = creatTree(data)

    search = searchList.reverse.distinct

  }

  /**
    *
    * @param prediction
    * @param dtnode
    */
  def predict(prediction: Dataset[Row]) = {

    val predictRdd = prediction.rdd.map(row => {
      def finder(node: DtreeNode, flist: List[String]): String = {
        val ftsValue: String = row.getAs[String](flist.head)
        node.label match {
          case "" =>
            val nextNode: DtreeNode = node.nexts.find(_._1 == ftsValue).get._2
            finder(nextNode, flist.tail)
          case _ => node.label
        }

      }
      val res: String = finder(node, search)
      Row.merge(row, Row(res))
    })

    val predictDfSchema =
      prediction.schema.add(StructField("predict", StringType))

    spark.createDataFrame(predictRdd, predictDfSchema)

  }

}
