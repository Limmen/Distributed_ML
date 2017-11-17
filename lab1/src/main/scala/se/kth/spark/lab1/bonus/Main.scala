package se.kth.spark.lab1.bonus

import ch.systemsx.cisd.hdf5.{ HDF5CompoundDataMap, HDF5FactoryProvider, IHDF5Reader, IHDF5SimpleReader, IHDF5Writer }
import java.io.File
import org.apache.spark._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SQLContext
import se.kth.spark.lab1.task2.{Main => MainTask2}
import ch.systemsx.cisd.hdf5.IHDF5Reader


object Main {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val filePath = "src/main/resources/MillionSongSubset/data/A/A/A/TRAAABD128F429CF47.h5"
    val reader = HDF5FactoryProvider.get().openForReading(new File(filePath))
    if(reader.exists("/musicbrainz/songs")){
    val musicBrainzSongs = reader.readCompound("/musicbrainz/songs", classOf[HDF5CompoundDataMap])
    val year = musicBrainzSongs.get("year")
      println("year: " + year)
    }
  }
}
