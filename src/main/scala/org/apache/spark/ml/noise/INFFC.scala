package org.apache.spark.ml.noise

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.{Params, Param, ParamMap}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.knn.KNN
import org.apache.spark.sql.expressions.UserDefinedFunction

trait INFFCParams extends Params with HasLabelCol with HasFeaturesCol {
    val votingSchema = new Param[VotingSchema.VotingSchema](this, "votingSchema", "Voting schema")
    def getVotingSchema: VotingSchema.VotingSchema = $(votingSchema)
    setDefault(votingSchema, VotingSchema.Consensus)

    val classifiers = new Param[Array[Estimator[_]]](this, "classifiers", "Array of classifiers to use")
    def getClassifiers: Array[Estimator[_]] = $(classifiers)

    val K = new Param[Int](this, "K", "Number of neighbors")
    def getK: Int = $(K)
    setDefault(K, 5)

    val noiseScoreThreshold = new Param[Double](this, "noiseScoreThreshold", "Noise score threshold")
    def getNoiseScoreThreshold: Double = $(noiseScoreThreshold)
    setDefault(noiseScoreThreshold, 0.0)

    val stopCriteriaG = new Param[Int](this, "stopCriteriaG", "Number of consecutive iterations where the number of noisy examples is less than stopCriteriaP of the size of the original dataset")
    def getStopCriteriaG: Int = $(stopCriteriaG)
    setDefault(stopCriteriaG, 3)

    val stopCriteriaP = new Param[Double](this, "stopCriteriaP", "Percentage of the size of the original dataset")
    def getStopCriteriaP: Double = $(stopCriteriaP)
    setDefault(stopCriteriaP, 0.01)
}

class INFFC(override val uid: String) extends Transformer with INFFCParams {
    def this() = this(Identifiable.randomUID("INFFC"))
    
    def setLabelCol(value: String): this.type =  set(labelCol, value)
    def setFeaturesCol(value: String): this.type =  set(featuresCol, value)
    def setClassifiers(value: Array[Estimator[_]]): this.type = set(classifiers, value)
    def setVotingSchema(value: VotingSchema.VotingSchema): this.type = set(votingSchema, value)
    def setK(value: Int): this.type = set(K, value)
    def setNoiseScoreThreshold(value: Double): this.type = set(noiseScoreThreshold, value)
    def setStopCriteriaG(value: Int): this.type = set(stopCriteriaG, value)
    def setStopCriteriaP(value: Double): this.type = set(stopCriteriaP, value)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val ensemble = new VoteEnsemble()
            .setVotingSchema($(votingSchema))
            .setClassifiers($(classifiers))
            .setLabelCol($(labelCol))

        val original_df = dataset.toDF()
        var current_df = original_df

        val stopCriteriaNoiseExamples = original_df.count() * $(stopCriteriaP)
        var stopCriteriaIters = 0

        while (stopCriteriaIters == $(stopCriteriaG)) {
            val noise_free_df = preliminaryNoiseFilter(ensemble, current_df)
            val filtered_df = noiseFreeFiltering(ensemble, current_df, noise_free_df)
            
            val n_noisy_examples = current_df.count() - filtered_df.count()
            if (n_noisy_examples < stopCriteriaNoiseExamples) {
                stopCriteriaIters += 1
            } else {
                stopCriteriaIters = 0
            }

            current_df = filtered_df
        }

        current_df
    }

    private def preliminaryNoiseFilter(
            vote_ensemble: VoteEnsemble,
            dataset: DataFrame): DataFrame = {    
        val vote_result = vote_ensemble.fit(dataset).transform(dataset)
        vote_result.filter("noisy == false")
    }

    private def noiseFreeFiltering(
            vote_ensemble: VoteEnsemble,
            dataset: DataFrame,                    
            noise_free_df: DataFrame): DataFrame = {
        val vote_result = vote_ensemble.fit(noise_free_df).transform(dataset)

        val knn = new KNN()
            .setTopTreeSize(10)
            .setFeaturesCol($(featuresCol))
            .setAuxCols(Array($(featuresCol), $(labelCol), "noisy"))

        // K+1 because KNN returns the sample itself as a neighbour,
        // we will discard it afterwards.
        // See: https://github.com/saurfang/spark-knn/issues/7
        val knn_model = knn.fit(vote_result).setK($(K)+1)
        val knn_result = knn_model.transform(vote_result)

        val computeNoiseScoreUDF = getComputeNoiseScoreUDF()
        val df_with_noiseScore = knn_result.withColumn(
            "noise_score", computeNoiseScoreUDF(col("neighbors"), col("noisy"))
        )
        
        df_with_noiseScore.filter(s"noise_score <= ${$(noiseScoreThreshold)}")
            .drop("neighbors", "noisy", "noise_score")
    }

    private def getComputeNoiseScoreUDF = udf {
        (neighbors: Array[Double], noisy: Boolean) => {
            neighbors.takeRight($(K))
        }
    }

    override def transformSchema(schema: StructType): StructType = schema
    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}