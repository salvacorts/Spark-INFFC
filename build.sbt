val sparkVersion = "3.1.2"

lazy val root = (project in file("."))
    .settings(
        name := "Spark-INFFC",
        organization := "com.github.salvacorts",
        version := "0.0.1",
        scalaVersion := "2.12.14",

        libraryDependencies ++= Seq(
            "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
            "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",

            "org.scalatest" %% "scalatest" % "3.2.9" % "test",

            "com.github.salvacorts" %% "spark-knn" % "0.3.1"
        )
    )