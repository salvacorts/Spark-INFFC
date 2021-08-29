import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame}

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.LinearSVC

import org.apache.spark.ml.noise.VotingSchema
import org.apache.spark.ml.noise.RandomNoise
import org.apache.spark.ml.noise.INFFC

import org.apache.log4j.{Logger, Level}

object Benchmark {
    def fitAndEvaluate(preffix: String,
                       model_name: String,
                       estimator: Estimator[_],
                       df_train: DataFrame,
                       df_test: DataFrame,
                       evaluator: MulticlassClassificationEvaluator) {
        val accuracy_evaluator = evaluator.setMetricName("accuracy")
        val f1_evaluator = evaluator.setMetricName("f1")

        val model = estimator.fit(df_train).asInstanceOf[Model[_]]

        val train_predictions = model.transform(df_train)
        val test_predictions = model.transform(df_test)

        val train_accuracy = accuracy_evaluator.evaluate(train_predictions)
        val test_accuracy = accuracy_evaluator.evaluate(test_predictions)
        val train_f1 = f1_evaluator.evaluate(train_predictions)
        val test_f1 = f1_evaluator.evaluate(test_predictions)

        println(s"$preffix $model_name train: Accuracy=$train_accuracy -- F1=$train_f1")
        println(s"$preffix $model_name test: Accuracy=$test_accuracy -- F1=$test_f1")
    }

    def evaluate(preffix: String, train_df: DataFrame, test_df: DataFrame) {
        val tree = new DecisionTreeClassifier()
            .setLabelCol("class")
            .setFeaturesCol("features")
            .setMaxDepth(20)
        
        val log = new LogisticRegression()
            .setLabelCol("class")
            .setFeaturesCol("features")

        val svm = new LinearSVC()
            .setLabelCol("class")
            .setPredictionCol("prediction")
            .setMaxIter(10)
            .setRegParam(0.1)

        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("class")
            .setPredictionCol("prediction")

        fitAndEvaluate(preffix, "DT", tree, train_df, test_df, evaluator)
        fitAndEvaluate(preffix, "LOG", log, train_df, test_df, evaluator)
        fitAndEvaluate(preffix, "SVM", svm, train_df, test_df, evaluator)
    }

    def addNoise(df: DataFrame, percentage: Double): DataFrame = {
        val random_noise = new RandomNoise()
            .setLabelCol("class")
            .setNoisePercentage(percentage)

        random_noise.transform(df)
    }

    def runINFFC(noise_percentage: Double, train_df: DataFrame, test_df: DataFrame): DataFrame = {
        Logger.getLogger("org.apache.spark.ml.noise").setLevel(Level.DEBUG)

        evaluate("Original", train_df, test_df)

        val noisy_train_df = addNoise(train_df, noise_percentage)
        evaluate("Noisy", noisy_train_df, test_df)

        val classifiers = Array[Estimator[_]](
            new NaiveBayes()
                .setLabelCol("class")
                .setFeaturesCol("features")
                .setModelType("gaussian"),

            new RandomForestClassifier()
                .setLabelCol("class")
                .setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(8),

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

        val clean_train_df = infcc.transform(noisy_train_df)
        println(s"Initial df size: ${noisy_train_df.count()}")
        println(s"Clean df size: ${clean_train_df.count()}")

        evaluate("Clean", clean_train_df, test_df)

        clean_train_df
    }

    // TODO: Add Diego methods...
}