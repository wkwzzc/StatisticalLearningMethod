package com.wk.algorithms


import breeze.linalg.{DenseMatrix, DenseVector, Transpose}

import scala.collection.mutable.ListBuffer

/**
 * https://www.cnblogs.com/gongyanzh/p/12880387.html
 *
 * @param pi
 * @param stateTransitionMatrix
 * @param confusionMatrix
 */
case class hmm(
                pi: DenseVector[Double],
                stateTransitionMatrix: DenseMatrix[Double],
                confusionMatrix: DenseMatrix[Double]
              ) {

  val n: Int = stateTransitionMatrix.cols

  def getDistDate(dist: DenseVector[Double]): Int = {
    var initState: Int = 0
    for (i <- 0 until dist.length) {
      if (math.random <= dist.slice(0, i).toArray.sum) {
        initState = i
        return initState
      }
    }
    initState
  }

  def generate(t: Int) = {
    //  require(true)
    // 根据初试概览向量随机生成初始状态
    val initState: Int = getDistDate(pi)
    // 生成第一个观测
    val inner = confusionMatrix(initState, ::).inner
    val initDate: Int = getDistDate(inner)

    //生成余下的状态和序列
    val datas = new ListBuffer[Int]
    datas.append(initDate)

    for (i <- 1 until t) {
      val st = getDistDate(stateTransitionMatrix(initState, ::).inner)
      datas append getDistDate(confusionMatrix(st, ::).inner)

    }
    datas.toArray

  }


  /**
   * 前向算法
   *
   * @param x 观测序列
   * @return
   */
  def computProb(x: DenseVector[Int]) = {


    //    step1：初始化 𝛼𝑖(1)=𝜋𝑖∗𝑏𝑖(𝑂1)
    //    step2：计算 𝛼𝑖(𝑡)=(∑𝑁𝑗=1𝛼𝑗(𝑡−1)𝑎𝑗𝑖)𝑏𝑖(𝑂𝑡)
    //    step3：𝑃(𝑂|𝜆)=∑𝑁𝑖=1𝛼𝑖(𝑇)
    // Step1 计算初值   𝛼𝑖(1)=𝜋𝑖∗𝑏𝑖(𝑂1)
    val b1 = confusionMatrix(::, x(0)).toArray
    val alpha1: Array[Double] = pi.toArray.zip(b1).map(x => x._1 * x._2)


    //    for t in range(1,T):
    //    for i in range(N):
    //      temp = 0
    //    for j in range(N):
    //      temp += alpha[j][t-1]*A[j][i]
    //    alpha[i][t] = temp*B[i][O[t]]
    //Step2 递推计算 alpha(t)  𝛼𝑖(𝑡)=(∑𝑁𝑗=1𝛼𝑗(𝑡−1)𝑎𝑗𝑖)𝑏𝑖(𝑂𝑡)
    for (t <- 1 until x.length) {
      for (i <- 0 until n) {

      }
      val alphat = 0


    }
    // 终止计算(概率)
    //    #step3
    //    proba = 0
    //    for i in range(N):
    //      proba += alpha[i][-1]
    //    return proba,alpha


    alpha1
  }

}

object hmm {


  def main(args: Array[String]): Unit = {

    //    val pi = DenseVector(Array.fill(4)(0.25))
    //    val stateTransitionMatrix = DenseMatrix((0.0, 1.0, 0.0, 0.0), (0.4, 0.0, 0.6, 0.0), (0.0, 0.4, 0.0, 0.6), (0.0, 0.0, 0.5, 0.5))
    //    val confusionMatrix = DenseMatrix((0.5, 0.5), (0.3, 0.7), (0.6, 0.4), (0.8, 0.2))


    val pi = DenseVector(Array(0.2, 0.4, 0.4))
    val stateTransitionMatrix = DenseMatrix((0.5, 0.2, 0.3), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5))
    val confusionMatrix = DenseMatrix((0.5, 0.5), (0.7, 0.3), (0.4, 0.6))

    val hmmModel = hmm(pi, stateTransitionMatrix, confusionMatrix)

    val hmmData = hmmModel.generate(3)
    println(hmmData.mkString(","))
    //    for (i <- 0 to 100){
    //    println( s"第${i}次:" + hmmModel.generate(10).mkString(","))
    //    }
    val doubles = hmmModel.computProb(DenseVector(Array(0, 1, 0)))
    println(doubles.mkString(","))
  }

}