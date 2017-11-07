package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vector
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



    val polynomialExpansionT = new PolynomialExpansion()
      .setInputCol("nonpolynomial_features")
      .setOutputCol("features")
      .setDegree(2)
    val transformations = MainTask2.arrayOfFeatureTransforms(sqlContext, rawDF, "nonpolynomial_features") ++ Array(polynomialExpansionT)

    val myLR = new LinearRegression().setMaxIter(50).setRegParam(0.1).setElasticNetParam(0.1)

    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.regParam, Array(0.0001, 0.001, 0.01, 0.2, 0.5, 0.9))
      .addGrid(myLR.maxIter, Array(10, 25, 40, 60, 100, 200))
      .build()

    val pipeline: Pipeline = new Pipeline().setStages(transformations ++ Array(myLR))


    val cv: CrossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator("rmse"))
      .setEstimatorParamMaps(paramGrid)

    val cvModel = cv.fit(rawDF)

    val summary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(7).asInstanceOf[LinearRegressionModel].summary

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