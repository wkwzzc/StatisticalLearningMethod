package CH3_KNearestNeibor

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


/**
  * Created by WZZC on 2019/11/29
  **/
case class KnnModel(data: DataFrame, labelName: String) extends Serializable {

  private val spark = data.sparkSession

//  import spark.implicits._
  // 使用.rdd的时候不能使用 col
//  private val sfadsfaggaggsagafasavsa: String = UUID.randomUUID().toString

  private val ftsName: String = Identifiable.randomUID("KnnModel")

  // 数据特征名称
  private val fts: Array[String] = data.columns.filterNot(_ == labelName)

  val shapes: Int = fts.length

  def vec2Seq = udf((vec: DenseVector) => vec.toArray.toSeq)

  /**
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

  private val
  kdtrees: Array[TreeNode] = dataTransForm(data)
    .withColumn(ftsName, vec2Seq(col(ftsName)))
    .select(labelName, ftsName)
    .withColumn("partitionIn", spark_partition_id())
    .rdd //在大数据情况下，分区构建kdtree
    .map(row => {
      val partitionIn = row.getInt(2)
      val label = row.getString(0)
      val features = row.getAs[Seq[Double]](1)
      (partitionIn, label, features)
    })
    .groupBy(_._1)
    .mapValues(_.toSeq.map(tp3 => (tp3._2, tp3._3)))
    .mapValues(nn => TreeNode.creatKdTree(nn, 0, shapes))
    .values
    .collect()


  /**
    *
    * @param predictDf
    * @param k
    * @return
    */
  def predict(predictDf: DataFrame, k: Int): DataFrame = {

    // 此处方法重载需要注意:overloaded method needs result type
    def nsearchUdf = udf((seq: Seq[Double]) => predict(seq, k))

    dataTransForm(predictDf)
      .withColumn(ftsName, vec2Seq(col(ftsName)))
      .withColumn(labelName, nsearchUdf(col(ftsName)))
      .drop(ftsName)

  }

  /**
    *
    * @param predictData
    * @param k
    * @return
    */
  def predict(predictData: Seq[Double], k: Int): String = {

    // 查询的时候遍历每个kdtree，然后取结果集再排序
    val res: Array[(Double, Seq[Double], String)] = kdtrees
      .map(node => {
        TreeNode.knn(node, predictData, k)
          .map(tp2 => (tp2._1, tp2._2.value, tp2._2.label))
      })
      .flatMap(_.toSeq)
      .sortBy(_._1)
      .take(k)

    // 按照投票选举的方法选择分类结果
    val cl = res
      .map(tp3 => (tp3._3, 1))
      .groupBy(_._1)
      .mapValues(_.map(_._2).sum)
      .maxBy(_._2)
      ._1
    cl
  }


}
