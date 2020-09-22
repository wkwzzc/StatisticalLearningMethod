
object tfidf {

  def main(args: Array[String]): Unit = {



   def withScope()= {

     ///设置跟踪堆的轨迹的scope名字
     val ourMethodName: String = "withScope"


     // 获取当前线程的
     val callerMethodName: String = Thread.currentThread.getStackTrace()
       //移除前几个匹配断言函数的元素，移除该线程下的方法名不是ourMethodName（withScope）的前几个(以ourMethodName分割)元素
//       .dropWhile(_.getMethodName != ourMethodName)
       // 获取线程中方法名不是 ourMethodName（withScope）的方法名
       .find(_.getMethodName != ourMethodName)
       .map(_.getMethodName)  // 最终得到的结果 就是withScope
         .getOrElse("error")


     println(callerMethodName)
     callerMethodName
   }


    withScope()

  }

}
