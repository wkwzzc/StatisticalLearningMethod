package statisticslearn.DataUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.udf

/**
  * Created by WZZC on 2020/5/2
  **/
object udfFuns {


  /**
  * s数据类型转换  DenseVector -> Array
    * @return
    */
  def vec2Array = udf((vec: DenseVector) => vec.toArray)


}
