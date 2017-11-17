name := "lab1"

organization := "se.kth.spark"

version := "1.0"

scalaVersion := "2.11.1"

sparkVersion := "2.0.1"

javaOptions in run += "-Xms4048m -Xmx8048m -XX:ReservedCodeCacheSize=512m -XX:MaxMetaspaceSize=2048m"

//resolvers += Resolver.mavenLocal
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3"
libraryDependencies += "se.kth.spark" %% "lab1_lib" % "1.0-SNAPSHOT"
libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
libraryDependencies += "net.sf.opencsv" % "opencsv" % "2.3"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3" % "provided"
//libraryDependencies += "ch.systemsx.cisd" % "sis-base" % "1.0"
//libraryDependencies += "ch.systemsx.cisd" % "sis-jhdf5-batteries_included" % "1.0"
//unmanagedJars += file("lib/sis-jhdf5-batteries_included.jar")
unmanagedBase := baseDirectory.value / "lib"
spDependencies += "LLNL/spark-hdf5:0.0.4"

mainClass in assembly := Some("se.kth.spark.lab1.task7.Main")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
