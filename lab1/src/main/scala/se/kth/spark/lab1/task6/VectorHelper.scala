package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{ Vector, Vectors }

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    if(v1.size != v2.size)
      throw new IllegalArgumentException("Dot product between vectors of different dimension is undefined")
    (0 to v1.size-1).map((i) => v1(i) * v2(i)).sum
  }

  def dot(v: Vector, s: Double): Vector = {
    Vectors.dense(v.toArray.map((element) => element*s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    if(v1.size != v2.size)
      throw new IllegalArgumentException("sum product between vectors of different dimension is undefined")
    Vectors.dense((0 to v1.size-1).map((i) => v1(i) + v2(i)).toArray)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    assert(size > 0)
    Vectors.dense((0 to size-1).map((_) => fillVal).toArray)
  }
}
