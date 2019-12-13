package statisticslearn.DataUtils

import breeze.linalg.DenseVector

import scala.math._

/**
  * Created by WZZC on 2018/10/18
  **/
object distanceUtils {

  /**
    *
    * @param p1
    * @param p2
    * @return Euclidean Distance
    */
  def euclidean(p1: Seq[Double], p2: Seq[Double]) = {
    require(p1.size == p2.size)

    val d = p1
      .zip(p2)
      .map(tp => pow(tp._1 - tp._2, 2))
      .sum

    sqrt(d)
  }

  /**
    *
    * @param p1
    * @param p2
    * @return Manhattan Distance	
    */
  def manhattan(p1: Seq[Double], p2: Seq[Double]) = {
    require(p1.size == p2.size)

    p1.zip(p2).map(tp => abs(tp._1 - tp._2)).sum

  }

  /**
    *
    * @param p1
    * @param p2
    * @return Chebyshev Distance
    */
  def chebyshev(p1: Seq[Double], p2: Seq[Double]) = {

    require(p1.size == p2.size)

    p1.zip(p2).map(tp => abs(tp._1 - tp._2)).max

  }

  /**
    *
    * @param p1
    * @param p2
    * @param p
    * @return Minkowski Distance
    */
  def minkowski(p1: Seq[Double], p2: Seq[Double], p: Int) = {
    require(p1.size == p2.size)
    val d = p1
      .zip(p2)
      .map(tp => pow(abs(tp._1 - tp._2), p))
      .sum

    pow(d, 1 / p)
  }

  /**
    *
    * @param p1
    * @param p2
    * @return 标准化欧式距离
    */
  def standardizedEuclidean(p1: Seq[Double], p2: Seq[Double]) = {
    require(p1.size == p2.size)
    var distant = 0d
    for (i <- 0 until p1.length) {
      // 计算分量 i 的标准差
      val mean = (p1(i) + p2(i)) / 2
      val theta = sqrt((pow((p1(i) - mean), 2) + pow((p2(i) - mean), 2)) / 2)
      distant += pow((p1(i) + p2(i)) / theta, 2)
    }
    sqrt(distant)
  }

  /**
    *
    * @param p1
    * @param p2
    * @return Hamming Distance
    */
  @deprecated
  def hamming(p1: String, p2: String) = {
    var distant = 0d
    for (i <- 0 until p1.length) {
      if (p1(i) == p2(i)) distant += 1
    }
    distant
  }

  /**
    *
    * @param p1
    * @param p2
    * @return Jaccard Distance  d(A,B) = 1-J(A,B) 表示Jaccard相似系数
    */
  def jaccard(p1: Seq[Any], p2: Seq[Any]) = {

    val anb = p1.intersect(p2).distinct
    val aub = p1.union(p2).distinct

    val jaccardcoefficient = anb.length.toDouble / aub.length

    1 - jaccardcoefficient

  }

  /**
    *
    * @param p1
    * @param p2
    * @return Tanimoto Distance
    */
  def tanimoto(p1: Seq[Double], p2: Seq[Double]) = {
    require(p1.size == p2.size)

    val v1 = new DenseVector(p1.toArray)
    val v2 = new DenseVector(p2.toArray)

    val a: Double = p1.map(pow(_, 2)).sum
    val b: Double = p2.map(pow(_, 2)).sum
    val pq = v1.dot(v2)

    pq / (a + b - pq)
  }

  /**
    *
    * @param p1
    * @param p2
    */
  def cos(p1: Seq[Double], p2: Seq[Double]) = {
    require(p1.size == p2.size)

    val v1 = new DenseVector(p1.toArray)
    val v2 = new DenseVector(p2.toArray)

    val a = sqrt(p1.map(pow(_, 2)).sum)
    val b = sqrt(p2.map(pow(_, 2)).sum)

    val ab = v1.t * v2

    ab / (a * b)
  }



}
