name := "lab1"

organization := "se.kth.spark"

version := "1.0"

scalaVersion := "2.11.1"

//resolvers += Resolver.mavenLocal
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3"
libraryDependencies += "se.kth.spark" %% "lab1_lib" % "1.0-SNAPSHOT"
libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

mainClass in assembly := Some("se.kth.spark.lab1.task6.Main")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
