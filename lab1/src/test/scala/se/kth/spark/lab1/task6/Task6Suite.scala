package se.kth.spark.lab1.task6

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.linalg.Vectors
import org.scalatest._
/**
 * Test suite for Task 6
 */
class Task6Suite extends FunSuite with Matchers with BeforeAndAfter {
  private var sc: SparkContext = _
  before {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    sc = new SparkContext(conf)
  }
  after {
    if (sc != null) {
      sc.stop()
    }
  }
  test("dot product test") {
    assert(VectorHelper.dot(v1 = Vectors.dense(1.0, 2.0, 3.0), v2 = Vectors.dense(4.0, -5.0, 6.0)) == 12.0)
    assert(VectorHelper.dot(v1 = Vectors.dense(-4.0, -9.0), v2 = Vectors.dense(-1.0, 2.0)) == -14.0)
    assert(VectorHelper.dot(v1 = Vectors.dense(1.0, 2.0), v2 = Vectors.dense(4, 8)) == 20)
    assert(VectorHelper.dot(v1 = Vectors.dense(1.0, 3.0, -5.0), v2 = Vectors.dense(4, -2, -1)) == 3)
    intercept[IllegalArgumentException] {
      VectorHelper.dot(v1 = Vectors.dense(1.0, 3.0, -5.0), v2 = Vectors.dense(4, -2))
    }
    assert(VectorHelper.dot(v = Vectors.dense(1.0, 2.0, 3.0), s = 4.0) == Vectors.dense(4.0, 8.0, 12.0))
    assert(VectorHelper.dot(v = Vectors.dense(1.0, 2.0, 3.0), s = 9.0) == Vectors.dense(9.0, 18.0, 27.0))
  }

  test("sum test") {
    assert(VectorHelper.sum(v1 = Vectors.dense(1.0, 2.0, 3.0), v2 = Vectors.dense(4.0, -5.0, 6.0)) == Vectors.dense(5.0, -3.0, 9.0))
    assert(VectorHelper.sum(v1 = Vectors.dense(-4.0, -9.0), v2 = Vectors.dense(-1.0, 2.0)) == Vectors.dense(-5.0, -7.0))
    assert(VectorHelper.sum(v1 = Vectors.dense(1.0, 2.0), v2 = Vectors.dense(4.0, 8.0)) == Vectors.dense(5.0, 10.0))
    assert(VectorHelper.sum(v1 = Vectors.dense(1.0, 3.0, -5.0), v2 = Vectors.dense(4.0, -2.0, -1.0)) == Vectors.dense(5.0, 1.0, -6.0))
    intercept[IllegalArgumentException] {
      VectorHelper.sum(v1 = Vectors.dense(1.0, 3.0, -5.0), v2 = Vectors.dense(4, -2))
    }
  }

  test("fill test") {
    assert(VectorHelper.fill(size = 3, fillVal = 1.0) == Vectors.dense(1.0, 1.0, 1.0))
    assert(VectorHelper.fill(size = 4, fillVal = 2.0) == Vectors.dense(2.0, 2.0, 2.0, 2.0))
  }

  test("rmse test") {
    assert(Math.floor(Helper.rmse(sc.parallelize(Seq((4.0, 1.0), (19.0, 2.0))))).toInt == 12.0)
    assert(Math.floor(Helper.rmse(sc.parallelize(Seq((145.0, 179.0), (45.0, 399.0))))).toInt == 251.0)
  }

  test("predict one test") {
    assert(Helper.predictOne(Vectors.dense(39.0, 245.0, 9.0), Vectors.dense(1.0, 2.0, 3.0)) == 556.0)
  }

  test("predict test") {
    val predicted = Helper.predict(Vectors.dense(39.0, 245.0, 9.0), sc.parallelize(Seq(Instance(600.0, Vectors.dense(1.0, 2.0, 3.0))))).collect()
    assert(predicted.length == 1)
    assert(predicted(0) == (600.0, 556.0))
  }
  /*
  test("gradient summand test"){
    val mlri = new MyLinearRegressionImpl("mlri")
    assert(mlri.gradientSummand(weights = Vectors.dense(39.0, 245.0, 9.0), lp = Instance(600.0, Vectors.dense(1.0,2.0,3.0))) == Vectors.dense(-44.0,-88.0,-132.0))
  }

  test("gradient test"){
    val mlri = new MyLinearRegressionImpl("mlri")
    assert(mlri.gradient(d = sc.parallelize(Seq(Instance(600.0, Vectors.dense(1.0,2.0,3.0)), Instance(480.0, Vectors.dense(13.0,1.0,1.0)))), Vectors.dense(39.0, 245.0, 9.0)) == Vectors.dense(3609.0, 193.0, 149.0))
  }
 */
}
