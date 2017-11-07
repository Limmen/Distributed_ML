package se.kth.spark.lab1.task2

import org.apache.spark.sql.functions.{max, min}
import se.kth.spark.lab1._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.{DataFrame, SQLContext}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw").cache()

    rawDF.show(5)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(arrayOfFeatureTransforms(sqlContext, rawDF, "features", Array(1, 2, 3)))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model
    val data = pipelineModel.transform(rawDF)

    //Step11: drop all columns from the dataframe other than label and features
    val cleanData = data.drop("raw", "parsed", "vector", "label", "label_double").toDF(Seq("label", "features"): _*)

    cleanData.show(5)
  }

  /**
   * Step 1-7 , extracted to own method for reuse in subsequent tasks.
   */
  def arrayOfFeatureTransforms(sqlContext: SQLContext, rawDf: DataFrame, featureColName: String, featureIndices: Array[Int]): Array[Transformer] = {
    import sqlContext.implicits._

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("parsed")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    //val transformed = regexTokenizer.transform(rawDF).drop("raw")
    //transformed.show(5)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    arr2Vect.setInputCol("parsed")
    arr2Vect.setOutputCol("vector")

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("label_vec")
    lSlicer.setIndices(Array(0))

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    // It is already double?
    val vec2Double = (vec: Vector) => vec(0)
    val v2d = new Vector2DoubleUDF(vec2Double)
    v2d.setInputCol("label_vec")
    v2d.setOutputCol("label_double")

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYear = rawDf.agg(min(rawDf.col("raw"))).map(r => r.getString(0).split(",")(0).toDouble).head
    val shift = (year: Double) => year - minYear
    val lShifter = new DoubleUDF(shift)
    lShifter.setInputCol("label_double")
    lShifter.setOutputCol("label")

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol(featureColName)
    fSlicer.setIndices(featureIndices)

    Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer)
  }
}
