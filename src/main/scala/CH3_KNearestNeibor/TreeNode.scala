package CH3_KNearestNeibor

/**
  * Created by WZZC on 2019/11/29
  **/
/**
  *
  * @param label 分类指标
  *  @param value 节点数据
  *  @param dim   当前切分维度
  *  @param left  左子节点
  *  @param right 右子节点
  */
case class TreeNode(label: String,
                    value: Seq[Double],
                    dim: Int,
                    var left: TreeNode,
                    var right: TreeNode)
    extends Serializable {}

object TreeNode {
  import statisticslearn.DataUtils.distanceUtils._

  /**
    *创建KD 树
    *
    * @param value
    * @param dim
    * @param shape
    * @return
    */
  def creatKdTree(value: Seq[(String, Seq[Double])],
                  dim: Int,
                  shape: Int): TreeNode = {

    // 数据按照当前划分的维度排序
    val sorted: Seq[(String, Seq[Double])] = value.sortBy(tp2 => tp2._2(dim))

    //中间位置的索引
    val midIndex: Int = value.length / 2

    sorted match {
      // 当节点为空时，返回null
      case Nil => null

      //节点不为空时，递归调用方法
      case _ =>
        val left: Seq[(String, Seq[Double])] = sorted.slice(0, midIndex)
        val right: Seq[(String, Seq[Double])] =
          sorted.slice(midIndex + 1, value.length)

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
  def knn(treeNode: TreeNode, data: Seq[Double], k: Int = 1) = {

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
