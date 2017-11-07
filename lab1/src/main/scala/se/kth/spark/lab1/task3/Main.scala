package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
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


    //val lrStage = ???
    val pipelineModel: PipelineModel = getLinearRegPipeline(sqlContext, rawDF)._2.fit(rawDF)

    val summary = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel].summary

    println(s"numIterations: ${summary.totalIterations}")
    println(s"RMSE: ${summary.rootMeanSquaredError}")
    println(s"r2: ${summary.r2}")

    pipelineModel.transform(rawDF).select("features","label","prediction").collect().take(5).foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"$features, $label) -> prediction=$prediction")
    }
  }

  def getLinearRegPipeline(sQLContext: SQLContext, rawDF: DataFrame): (LinearRegression, Pipeline) = {
    val myLR = new LinearRegression().setMaxIter(50).setRegParam(0.1).setElasticNetParam(0.1)
    (myLR, new Pipeline().setStages(MainTask2.arrayOfFeatureTransforms(sQLContext,rawDF, "features") ++ Array(myLR)))
  }
}