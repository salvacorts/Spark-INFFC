import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import util.Random
import org.apache.log4j.{Logger, Level}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.classification.NaiveBayes

import org.apache.spark.ml.noise.VotingSchema
import org.apache.spark.ml.noise.RandomNoise
import org.apache.spark.ml.noise.INFFC

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Main {
    def main(args: Array[String]) = {
        val spark = SparkSession
            .builder()
            .master(sys.env("SPARK_MASTER"))
            .appName("Spark-INFCC SUSY")
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        val susy_train_path = args(0)
        val susy_test_path = args(1)
        val noise_percentage = args(2).toDouble

        println(s"Noise percentage: $noise_percentage")
        println(s"Loading susy dataset from:")
        println(s"\tTrain: $susy_train_path")
        println(s"\tTest: $susy_test_path")

        val schema = new StructType()
            .add("one", DoubleType)
            .add("two", DoubleType)
            .add("three", DoubleType)
            .add("four", DoubleType)
            .add("five", DoubleType)
            .add("six", DoubleType)
            .add("seven", DoubleType)
            .add("eight", DoubleType)
            .add("nine", DoubleType)
            .add("ten", DoubleType)
            .add("eleven", DoubleType)
            .add("twelve", DoubleType)
            .add("thirdteen", DoubleType)
            .add("fourteen", DoubleType)
            .add("fifteen", DoubleType)
            .add("sixteen", DoubleType)
            .add("seventeen", DoubleType)
            .add("eighteen", DoubleType)
            .add("class", DoubleType)

        val train_df = spark.read
            .options(Map("inferSchema" -> "false",
                         "delimiter" -> ",",
                         "header" -> "false"))
            .schema(schema)
            .csv(susy_train_path)

        val test_df = spark.read
            .options(Map("inferSchema" -> "false",
                         "delimiter" -> ",",
                         "header" -> "false"))
            .schema(schema)
            .csv(susy_test_path)

        println(s"SUSY Train size: ${train_df.count()} instances")
        println(s"SUSY Test size: ${test_df.count()} instances")

        val vector_assembler = new VectorAssembler()
            .setInputCols(train_df.columns.take(train_df.columns.length-1))
            .setOutputCol("features")

        val train_df_ml = vector_assembler.transform(train_df)
        val test_df_ml = vector_assembler.transform(test_df)

        val random_noise = new RandomNoise()
            .setLabelCol("class")
            .setNoisePercentage(noise_percentage)

        val noisy_train_df = random_noise.transform(train_df_ml)
        noisy_train_df.cache()
        
        val tree = new DecisionTreeClassifier()
            .setLabelCol("class")
            .setFeaturesCol("features")
            .setMaxDepth(20)

        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("class")
            .setPredictionCol("prediction")
        val accuracy_evaluator = evaluator.setMetricName("accuracy")
        val f1_evaluator = evaluator.setMetricName("f1")

        val noisy_tree_model = tree.fit(noisy_train_df)
        val noisy_train_predictions = noisy_tree_model.transform(noisy_train_df)
        val noisy_test_predictions = noisy_tree_model.transform(test_df_ml)

        val noisy_train_accuracy = accuracy_evaluator.evaluate(noisy_train_predictions)
        val noisy_test_accuracy = accuracy_evaluator.evaluate(noisy_test_predictions)
        val noisy_train_f1 = f1_evaluator.evaluate(noisy_train_predictions)
        val noisy_test_f1 = f1_evaluator.evaluate(noisy_test_predictions)
        println(s"Noisy train: Accuracy=$noisy_train_accuracy -- F1=$noisy_train_f1")
        println(s"Noisy test: Accuracy=$noisy_test_accuracy -- F1=$noisy_test_f1")

        val classifiers = Array[Estimator[_]](
            new NaiveBayes()
                .setLabelCol("class")
                .setFeaturesCol("features")
                .setModelType("gaussian"),

            new RandomForestClassifier()
                .setLabelCol("class")
                .setFeaturesCol("features")
                .setNumTrees(10)
                .setMaxDepth(12),

            new LogisticRegression()
                .setLabelCol("class")
                .setFeaturesCol("features")
        )

        val infcc = new INFFC()
            .setVotingSchema(VotingSchema.Majority)
            .setClassifiers(classifiers)
            .setLabelCol("class")
            .setFeaturesCol("features")
            .setK(5)
            .setNoiseScoreThreshold(0.0)
            .setStopCriteriaG(3)
            .setStopCriteriaP(0.01)

        Logger.getLogger("INFFC").setLevel(Level.DEBUG)
        Logger.getLogger("VoteEnsemble").setLevel(Level.DEBUG)

        val clean_train_df = infcc.transform(noisy_train_df)
        println(s"Initial df size: ${noisy_train_df.count()}")
        println(s"Clean df size: ${clean_train_df.count()}")

        val clean_tree_model = tree.fit(clean_train_df)
        val clean_train_predictions = clean_tree_model.transform(clean_train_df)
        val clean_test_predictions = clean_tree_model.transform(test_df_ml)

        val clean_train_accuracy = evaluator.evaluate(clean_train_predictions)
        val clean_test_accuracy = evaluator.evaluate(clean_test_predictions)
        val clean_train_f1 = f1_evaluator.evaluate(clean_train_predictions)
        val clean_test_f1 = f1_evaluator.evaluate(clean_test_predictions)
        println(s"Clean train: Accuracy=$clean_train_accuracy -- F1=$clean_train_f1")
        println(s"Clean test: Accuracy=$clean_test_accuracy -- F1=$clean_test_f1")

        spark.close()
    }
}