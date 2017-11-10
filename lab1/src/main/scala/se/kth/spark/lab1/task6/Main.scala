package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.apache.spark.ml.PipelineModel
import se.kth.spark.lab1.task2.{Main => MainTask2}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw").cache()
    //Our own implementation of LR estimator
    val pipelineModel: PipelineModel = getLinearRegPipeline(sqlContext, rawDF)._2.fit(rawDF)
    val lrStage = 6
    //Use our own model
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]
    val predsAndLabels = pipelineModel.transform(rawDF).select("features", "label", "prediction")
    val rmse = Helper.rmse(predsAndLabels.select("label", "prediction").rdd.map(row => (row.getAs[Double](0), row.getAs[Double](1))))
    println(s"RMSE: $rmse")
    predsAndLabels.collect().take(5).foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"$features, $label) -> prediction=$prediction")
    }
  }

  /**
   * Extract pipeline creation to method so it can be reused in subsequent steps
   */
  def getLinearRegPipeline(sQLContext: SQLContext, rawDF: DataFrame): (MyLinearRegressionImpl, Pipeline) = {
    val myLr = new MyLinearRegressionImpl()
    //Reuse feature transformations from task2
    (myLr, new Pipeline().setStages(MainTask2.arrayOfFeatureTransforms(sQLContext, rawDF, "features", Array(1, 2, 3)) ++ Array(myLr)))
  }
}
