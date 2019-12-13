package CH3_KNearestNeibor

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import statisticslearn.DataUtils.distanceUtils._
/**
  * Created by WZZC on 2019/11/29
  **/
case class KnnModel(data: DataFrame, labelName: String) extends Serializable {

  private val spark = data.sparkSession
  import spark.implicits._

  // 使用.rdd的时候不能使用 col
//  private val sfadsfaggaggsagafasavsa: String = UUID.randomUUID().toString
//  private val featuresCol =   col(sfadsfaggaggsagafasavsa)
//  println(sfadsfaggaggsagafasavsa)

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
      .setOutputCol("sfadsfaggaggsagafasavsa")
      .transform(dataFrame)
      .withColumn(
        "sfadsfaggaggsagafasavsa",
        vec2Seq($"sfadsfaggaggsagafasavsa")
      )

  }

  private val kdtrees = dataTransForm(data)
    .select(labelName, "sfadsfaggaggsagafasavsa")
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
    .mapValues(nn => creatKdTree(nn, 0, shapes))
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
      .withColumn("labelName", nsearchUdf($"sfadsfaggaggsagafasavsa"))
      .drop("sfadsfaggaggsagafasavsa")

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
        knn(node, predictData, k)
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

  /**
    *
    * @param value
    * @param dim
    * @param shape
    * @return
    */
  def creatKdTree(value: Seq[(String, Seq[Double])],
                  dim: Int,
                  shape: Int = shapes): TreeNode = {

    // 数据按照当前划分的维度排序
    val sorted: Seq[(String, Seq[Double])] =
      value.sortBy(tp2 => tp2._2(dim))
    //中间位置的索引
    val midIndex: Int = value.length / 2

    sorted match {
      // 当节点为空时，返回null
      case Nil => null
      //节点不为空时，递归调用方法
      case _ =>
        val left = sorted.slice(0, midIndex)
        val right = sorted.slice(midIndex + 1, value.length)

        val leftNode = creatKdTree(left, (dim + 1) % shape, shape) //左子节点递归创建树
        val rightNode = creatKdTree(right, (dim + 1) % shape, shape) //右子节点递归创建树

        TreeNode(
          sorted(midIndex)._1,
          sorted(midIndex)._2,
          dim,
          leftNode,
          rightNode
        )

    }
  }

  /**
    * 从root节点开始，DFS搜索直到叶子节点，同时在stack中顺序存储已经访问的节点。
    * 如果搜索到叶子节点，当前的叶子节点被设为最近邻节点。
    * 然后通过stack回溯:
    * 如果当前点的距离比最近邻点距离近，更新最近邻节点.
    * 然后检查以最近距离为半径的圆是否和父节点的超平面相交.
    * 如果相交，则必须到父节点的另外一侧，用同样的DFS搜索法，开始检查最近邻节点。
    * 如果不相交，则继续往上回溯，而父节点的另一侧子节点都被淘汰，不再考虑的范围中.
    * 当搜索回到root节点时，搜索完成，得到最近邻节点。
    *
    * @param treeNode
    * @param data
    * @param k
    * @return
    */
  def knn(treeNode: TreeNode, data: Seq[Double], k: Int) = {

//    implicit def vec2Seq(a:DenseVector[Double])=a.toArray.toSeq

    var resArr = new Array[(Double, TreeNode)](k)
      .map(_ => (Double.MaxValue, null))
      .asInstanceOf[Array[(Double, TreeNode)]]

    def finder(treeNode: TreeNode): TreeNode = {

      if (treeNode != null) {
        val dimr = data(treeNode.dim) - treeNode.value(treeNode.dim)
        if (dimr > 0) finder(treeNode.right) else finder(treeNode.left)

        val distc: Double = euclidean(treeNode.value, data)

        if (distc < resArr.last._1) {
          resArr.update(k - 1, (distc, treeNode))
          resArr = resArr.sortBy(_._1)
        }

        if (math.abs(dimr) < resArr.last._1)
          if (dimr > 0) finder(treeNode.left) else finder(treeNode.right)

      }
      resArr.last._2
    }

    finder(treeNode)
    resArr

  }

}
