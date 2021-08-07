import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.KNNClassifier
import scala.collection.mutable
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Model
import org.apache.spark.ml.Estimator

import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.noise.VoteEnsemble
import org.apache.spark.ml.noise.VotingSchema
import org.apache.spark.ml.noise.INFFC
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Main {
    val K = 5

    def main(args: Array[String]) {
        val spark = SparkSession.builder.appName("HelloWorld Application").getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        println(s"Hello World")

        val df = spark.read
            .options(Map("inferSchema" -> "true",
                         "delimiter" -> ",",
                         "header" -> "true"))
            .csv("file:///workspace/data/iris.csv")
        df.printSchema()

        val assembler = new VectorAssembler()
            .setInputCols(Array("sepal_length",
                                "sepal_width",
                                "petal_length",
                                "petal_width"))
            .setOutputCol("features")

        val indexer = new StringIndexer()
            .setInputCol("variety")
            .setOutputCol("varietyInt")

        var df_ml = assembler.transform(df)
        df_ml = indexer.fit(df_ml).transform(df_ml)
        df_ml.show()

        val classifiers = Array[Estimator[_]](
            new DecisionTreeClassifier()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features"),

            new RandomForestClassifier()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features")
                .setNumTrees(10),

            new LogisticRegression()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features")
        )

        val infcc = new INFFC()
            .setVotingSchema(VotingSchema.Majority)
            .setClassifiers(classifiers)
            .setLabelCol("varietyInt")
            .setFeaturesCol("features")
            .setK(5)
            .setNoiseScoreThreshold(0.0)
            .setStopCriteriaG(3)
            .setStopCriteriaP(0.01)

        val infcc_logger = Logger.getLogger(infcc.getClass().getName())
        infcc_logger.setLevel(Level.DEBUG)

        val clean_df = infcc.transform(df_ml)

        println(s"Initial df size: ${df_ml.count()}")
        println(s"Clean df size: ${clean_df.count()}")

        spark.stop()
    }

    def mainOld(args: Array[String]) {
        val spark = SparkSession.builder.appName("HelloWorld Application").getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        println(s"Hello World")

        val df = spark.read
            .options(Map("inferSchema" -> "true",
                         "delimiter" -> ",",
                         "header" -> "true"))
            .csv("file:///workspace/data/iris.csv")
        df.printSchema()

        val assembler = new VectorAssembler()
            .setInputCols(Array("sepal_length",
                                "sepal_width",
                                "petal_length",
                                "petal_width"))
            .setOutputCol("features")

        val indexer = new StringIndexer()
            .setInputCol("variety")
            .setOutputCol("varietyInt")

        var df_ml = assembler.transform(df)
        df_ml = indexer.fit(df_ml).transform(df_ml)
        df_ml.show()

        println("\n\n--- Voting Ensemble ---")
        val classifiers = Array[Estimator[_]](
            new DecisionTreeClassifier()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features"),

            new RandomForestClassifier()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features")
                .setNumTrees(10),

            new LogisticRegression()
                .setLabelCol("varietyInt")
                .setFeaturesCol("features")
        )

        val ensemble = new VoteEnsemble()
            .setVotingSchema(VotingSchema.Majority)
            .setClassifiers(classifiers)
            .setLabelCol("varietyInt")

        val ensemble_model = ensemble.fit(df_ml)
        val vote_result = ensemble_model.transform(df_ml)
        vote_result.printSchema()
        vote_result.show()

        vote_result.collect.foreach(sample => {
            val original_features = sample.getAs[Vector]("features")
            val original_variety = sample.getAs[Double]("varietyInt")
            val predictions = sample.getAs[mutable.WrappedArray[Double]]("predictions")
            val veredict = sample.getAs[Boolean]("noisy")

            println(s"\nSample: $original_features - $original_variety")
            println(s"predictions: $predictions - Clean: $veredict")
        })

        println(s"Initial df size: ${vote_result.count()}")
        val clean_set = vote_result.filter("noisy == false")
        println(s"Preliminary clean df size: ${clean_set.count()}")

        val knn = new KNN()
            .setTopTreeSize(10)
            .setFeaturesCol("features")
            .setAuxCols(Array("features", "varietyInt", "noisy"))

        // K+1 Since KNN returns the sample itself as a neighbour,
        // we will discard it. https://github.com/saurfang/spark-knn/issues/7
        val model = knn.fit(clean_set).setK(K+1)

        val results_df = model.transform(clean_set)
        results_df.printSchema()
        results_df.show()
        
        println("\n\n--- KNN ---")
        results_df.collect.foreach(sample => {
            val original_features = sample.getAs[Vector]("features")
            val original_variety = sample.getAs[Double]("varietyInt")
            val original_noisy = sample.getAs[Boolean]("noisy")
            println(s"\nSample: $original_features - $original_variety - $original_noisy")

            val neighbors = sample.getAs[mutable.WrappedArray[Row]]("neighbors")
                .takeRight(K) // Skips the first one which is this same point

            neighbors.foreach(neighbor => {
                val features = neighbor.getAs[Vector]("features")
                val variety = neighbor.getAs[Double]("varietyInt")
                val noisy = neighbor.getAs[Boolean]("noisy")
                println(s"Neighbor: $features - $variety - $noisy")
            })
        })

        spark.stop()
    }
}