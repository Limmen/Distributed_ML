package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector
import se.kth.spark.lab1.task3.{Main => Task3Main}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw").cache()


    val (myLR,pipeline) = Task3Main.getLinearRegPipeline(sqlContext, rawDF)
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.regParam, Array(0.0001, 0.001, 0.01, 0.2, 0.5, 0.9))
      .addGrid(myLR.maxIter, Array(10, 25, 40, 60, 100, 200))
      .build()

    val cv: CrossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator("rmse"))
      .setEstimatorParamMaps(paramGrid)


    val cvModel = cv.fit(rawDF)

    val summary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel].summary

    //print rmse of our model
    //do prediction - print first k
    println(s"numIterations: ${summary.totalIterations}")
    println(s"RMSE: ${summary.rootMeanSquaredError}")
    println(s"r2: ${summary.r2}")

    cvModel.bestModel.transform(rawDF).select("features","label","prediction").collect().take(5).foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"$features, $label) -> prediction=$prediction")
    }


  }
}