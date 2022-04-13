package HHM

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.Stream

/**
 * 参考资料：
 * https://zhuanlan.zhihu.com/p/85454896
 * https://zhuanlan.zhihu.com/p/111899116
 * https://www.cnblogs.com/gongyanzh/p/12880387.html#%E5%89%8D%E5%90%91%E7%AE%97%E6%B3%95
 *
 * @param pi                    隐状态初始概率分布
 * @param stateTransitionMatrix 状态转移矩阵
 * @param confusionMatrix       观测状态生成矩阵
 */
case class hmm(
                pi: DenseVector[Double],
                stateTransitionMatrix: DenseMatrix[Double],
                confusionMatrix: DenseMatrix[Double]
              ) {

  val n: Int = stateTransitionMatrix.cols

  // 根据给定的概率分布随机返回数据
  def getDistData(dist: DenseVector[Double]): Int = {
    var initState: Int = 0
    for (i <- 0 until dist.length) {
      if (math.random <= dist.slice(0, i).toArray.sum) {
        initState = i
        return initState
      }
    }
    initState
  }

  //   根据给定的参数生成观测序列
  def generate(t: Int): Array[Int] = {
    //  require(true)
    // 根据初试概览向量随机生成初始状态
    val initState: Int = getDistData(pi)
    // 生成第一个观测
    val inner = confusionMatrix(initState, ::).inner
    val initData: Int = getDistData(inner)

    //生成余下的状态和序列
    val datas = new ListBuffer[Int]
    datas.append(initData)

    for (i <- 1 until t) {
      val st = getDistData(stateTransitionMatrix(initState, ::).inner)
      datas append getDistData(confusionMatrix(st, ::).inner)

    }
    datas.toArray

  }


  /**
   * 前向算法
   *
   * @param o 观测序列                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              观测序列
   * @return 产生观测序列的概率
   */
  def forwardAlgorithm(o: DenseVector[Int]) = {

    val alphat = new ListBuffer[Array[Double]]()

    // Step1 计算初值   α(i)=πi∗b(i)(𝑂(1))
    // 获取第一个观测的概率分布
    val b0 = confusionMatrix(::, o(0)).toArray
    val alpha0: Array[Double] = pi.toArray.zip(b0).map(x => x._1 * x._2)

    alphat.append(alpha0)
    //α(t)(i)= [∑ α(t-1)(i) a(j)(i)]*b(i)(o(t-1))
    for (t <- 1 until o.length) { // 观测序列长度T
      val alphaij = new ListBuffer[Double]()
      for (i <- 0 until n) {
        val value = stateTransitionMatrix(::, i).toArray
        val bi = confusionMatrix(i, o(t))
        alphaij.append(alphat(t - 1).zip(value).map(x => x._1 * x._2).map(_ * bi).sum)
      }
      alphat.append(alphaij.toArray)
    }
    // step3 终止计算(概率)：𝑃(𝑂|𝜆)=∑ a(t)(i)
    alphat(n - 1).sum
  }

}

object hmm {


  def main(args: Array[String]): Unit = {

    val pi = DenseVector(Array(0.2, 0.4, 0.4))

    val stateTransitionMatrix = DenseMatrix((0.5, 0.2, 0.3), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5))

    val confusionMatrix = DenseMatrix((0.5, 0.5), (0.4, 0.6), (0.7, 0.3))

    val hmmModel = hmm(pi, stateTransitionMatrix, confusionMatrix)

    val doubles = hmmModel.forwardAlgorithm(DenseVector(Array(0, 1, 0)))

    print(doubles)

    val fibs: Stream[Int] = 0 #:: fibs.scanLeft(1) {
      _ + _
    }
    //    val fibs:LazyList[BigInt] = 0 #::fibs.scanLeft(1){ _+_ }

    print(fibs.take(10))
  }

}