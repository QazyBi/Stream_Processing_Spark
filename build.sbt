name := "spark assignment"

version := "1.0"

scalaVersion := "2.12.10"
val sparkVersion = "3.0.1"


libraryDependencies ++= Seq("org.apache.spark" %% "spark-streaming" % "2.4.4",
  "org.apache.bahir" %% "spark-streaming-twitter" % "2.4.0",
  "org.apache.spark" %% "spark-core" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4",
  "org.apache.spark" %% "spark-sql" % "2.4.4")
