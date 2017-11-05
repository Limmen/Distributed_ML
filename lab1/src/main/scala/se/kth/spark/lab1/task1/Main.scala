package se.kth.spark.lab1.task1


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {

  case class Song(year: Int, feature1: Double, feature2: Double, feature3: Double)

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw").cache()
    rawDF.show(5)

    val rdd = sc.textFile(filePath)


    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x => x.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(record => Song(record(0).toInt, record(1).toDouble, record(2).toDouble, record(3).toDouble))

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()

    songsDf.printSchema()
  }
}