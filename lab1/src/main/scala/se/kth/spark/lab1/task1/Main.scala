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

    //For implicit conversions like converting RDDs to DataFrames
    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    //rawDF without explicit schema
    val rawDF = sc.textFile(filePath).toDF("raw").cache()

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rawDF.show(5)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x => x.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features
    val songsRdd = recordsRdd.map(record => Song(record(0).toDouble.toInt, record(1).toDouble, record(2).toDouble, record(3).toDouble))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()
    //DF with explicit schema
    songsDf.printSchema()

    // Questions
    // 1. How many songs there are in the Dataframe
    println(s"Number of songs in the df: ${songsDf.select("year").count()}")
    // 2. How many songs were released between the years 1998 and 2000?
    val filteredSongs = songsDf.filter(songsDf.col("year").between(1998, 2000)).count()
    println(s"Number of songs between 1998 and 2000: $filteredSongs")
    // 3. What is the min, max mean value of the year column?
    songsDf.describe("year").show()
    // 4. Show the number of songs per year between the years 2000 and 2010
    val filteredSongs2 = songsDf.filter(songsDf.col("year").between(2000, 2010)).count()
    println(s"Number of songs between 2000 and 2010: ${filteredSongs2}")
  }
}
