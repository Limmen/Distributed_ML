package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Row, SQLContext}
import se.kth.spark.lab1.task2.{Main => MainTask2}

/*
 *
 * Output when training with the smaller csv file and all 6 features:
 *
 * Start training (using all 6 features)
 * Training done in: 235.010298526 seconds = 4 minutes
 * numIterations: 201
 * RMSE: 15.48973145600878
 * r2: 0.4769790368839615
 *
 * Output when training with the larger csv file (500k songs) and all 6 features:
 *
 * Start training (using all 6 features)
 * Training done in: 3014.769056968 seconds = 50 minutes
 * numIterations: 201
 * RMSE: 10.304488266843789
 * r2: 0.1112677268884611
 *
 */
object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    //val rawDF = sc.textFile(filePath).toDF("raw").cache()
   // val filePath = "src/main/resources/million-song-all.txt"
    /*
     * Convert the larger dataset to proper format.. it was malformed with nested strings
     */
    val rawDF = sc.textFile(filePath).map(row => {
      val cols = row.split(",")
      if (cols(0).contains("\"")) {
        cols.map(c => {
          c.substring(1, c.size - 1).toDouble
        }).mkString(",")
      } else
        row
    }).toDF("raw").cache()
    /*
     * Transformation that expands feature vector into a polynomial space.
     * Uses n-degree combination of original dimensions.
     * E.g (x,y) -> (x, x * x, y, x * y, y * y)
     * This transformation generates 2 way interactions, e.g if
     * the relationship between x and label depends on y and vice verse then these types of
     * featuers can be useful for linear models. (In deep models these type of features can
     * be learned by the model)
     */
    val polynomialExpansionT = new PolynomialExpansion()
      .setInputCol("nonpolynomial_features")
      .setOutputCol("features")
      .setDegree(2)

    //Reuse transformations from task2 but use all featuers instead of just 3.
    val transformations = MainTask2.arrayOfFeatureTransforms(sqlContext, rawDF, "nonpolynomial_features", Array(1, 2, 3, 4, 5, 6)) ++ Array(polynomialExpansionT)

    //Build pipeline with cross validation
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

    //Run Cross-Validation and choose best parameters
    println("Start training (using all 6 features)")
    val start = System.nanoTime()
    val cvModel = cv.fit(rawDF)
    val end = System.nanoTime()
    val elapsedTime = end - start
    val elapsedSeconds = elapsedTime / 1000000000.0
    println(s"Training done in: ${elapsedSeconds} seconds")

    val summary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(7).asInstanceOf[LinearRegressionModel].summary

    //print rmse of our model
    //do prediction - print first k
    println(s"numIterations: ${summary.totalIterations}")
    println(s"RMSE: ${summary.rootMeanSquaredError}")
    println(s"r2: ${summary.r2}")

    //Print predictions
    cvModel.bestModel.transform(rawDF).select("features", "label", "prediction").collect().take(5).foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"$features, $label) -> prediction=$prediction")
    }
  }
}
