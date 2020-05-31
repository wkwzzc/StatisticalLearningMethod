package CH5_DecisionTree

/**
  * Created by WZZC on 2019/12/6
  **/
case class DtreeNode(
                      ftsName: String, //feature name
                      label: String,
                      nexts: Seq[(String, DtreeNode)] = Nil //feature value seq (feature values with next node)
                    ) {

}
