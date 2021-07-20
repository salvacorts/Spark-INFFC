import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.knn.KNN

import scala.collection.mutable

object Main {
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

        val knn = new KNN().
            setTopTreeSize(10).
            setAuxCols(Array("features"))

        val model = knn.fit(df_ml).setK(5)

        val results_df = model.transform(df_ml)
        results_df.printSchema()
        results_df.show()

        results_df.collect.foreach(row => {
            val original_features = row.getAs[Vector]("features")
            println("Features: " + original_features)

            val neighbors_features = row.getAs[mutable.WrappedArray[Row]]("neighbors")
                .map(_.getAs[Vector]("features"))
            neighbors_features.foreach(features => {
                println("Neighbor features: " + features)
            })
        })

        spark.stop()
    }
}