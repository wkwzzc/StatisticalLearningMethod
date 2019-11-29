package CH3_KNearestNeibor
 /**
  * Created by WZZC on 2019/11/29
  **/ /**
  *
  * @param value 节点数据
  * @param dim   当前切分维度
  * @param left  左子节点
  * @param right 右子节点
  */
case class TreeNode(label: String,
                    value: Seq[Double],
                    dim: Int,
                    var left: TreeNode,
                    var right: TreeNode) extends Serializable {




}
