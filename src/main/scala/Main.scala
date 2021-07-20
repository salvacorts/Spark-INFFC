import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.knn.KNN

import scala.collection.mutable

object Main {
    val K = 5

    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("HelloWorld Application").getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        println(s"Hello World")

        val df = spark.read.
            options(Map("inferSchema"->"true",
                         "delimiter"->",",
                         "header"->"true")).
            csv("file:///workspace/data/iris.csv")
        df.printSchema()

        val assembler = new VectorAssembler().
            setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).
            setOutputCol("features")

        val indexer = new StringIndexer().
            setInputCol("variety").
            setOutputCol("varietyInt")

        var df_ml = assembler.transform(df)
        df_ml = indexer.fit(df_ml).transform(df_ml)
        df_ml.show()

        val knn = new KNN()
            .setTopTreeSize(10)
            .setFeaturesCol("features")
            .setAuxCols(Array("features", "varietyInt"))

        // K+1 Since KNN returns the sample itself as a neighbour,
        // we will discard it. https://github.com/saurfang/spark-knn/issues/7
        val model = knn.fit(df_ml).setK(K+1)

        val results_df = model.transform(df_ml)
        results_df.printSchema()
        results_df.show()

        results_df.collect.foreach(sample => {
            val original_features = sample.getAs[Vector]("features")
            val original_variety = sample.getAs[Double]("varietyInt")
            println(s"\nSample: $original_features - $original_variety")

            val neighbors = sample.getAs[mutable.WrappedArray[Row]]("neighbors")
                .takeRight(K) // Skips the first one which is this same point

            neighbors.foreach(neighbor => {
                val features = neighbor.getAs[Vector]("features")
                val variety = neighbor.getAs[Double]("varietyInt")
                println(s"Neighbor: $features - $variety")
            })
        })

        spark.stop()
    }
}