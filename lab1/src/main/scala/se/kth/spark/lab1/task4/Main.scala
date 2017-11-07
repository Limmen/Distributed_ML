package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Row, SQLContext}
import se.kth.spark.lab1.task3.{Main => Task3Main}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw").cache()

    //Reuse Pipeline from task 3
    val (myLR, pipeline) = Task3Main.getLinearRegPipeline(sqlContext, rawDF)
    /*
     * 6x6 parameter grid, e.g total 36 settings of parameters.
     * Base values are 0.1 and 50, the params for tuning is 3 lower and 3 higher than base.
     */
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.regParam, Array(0.0001, 0.001, 0.01, 0.2, 0.5, 0.9))
      .addGrid(myLR.maxIter, Array(10, 25, 40, 60, 100, 200))
      .build()

    /*
     * .setEstimator decides the algorithm to tune
     * .setEvaluator decides the metric to measure how well a fitted model performed
     * .setEstimatorParammaps decides parameter grid to choose parameters from
     * .setNumfolds decides number of (train,test) pairs generated where each pair
     * uses 2/3 of data for training and 1/3 for testing
     */
    val cv: CrossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator("rmse"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)

    //Run Cross-Validation and choose best parameters
    println("Start training")
    val start = System.nanoTime()
    val cvModel = cv.fit(rawDF)
    val end = System.nanoTime()
    val elapsedTime = end - start
    val elapsedSeconds = elapsedTime / 1000000000.0
    println(s"Training done in: ${elapsedSeconds} seconds")

    val summary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel].summary

    //print rmse of our model
    //do prediction - print first k
    println(s"numIterations: ${summary.totalIterations}")
    println(s"RMSE: ${summary.rootMeanSquaredError}")
    println(s"r2: ${summary.r2}")

    //Predictions
    cvModel.bestModel.transform(rawDF).select("features", "label", "prediction").collect().take(5).foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"$features, $label) -> prediction=$prediction")
    }
  }
}
