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
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.knn.KNN
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.{Vector, Vectors}
import scala.collection.mutable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.BooleanType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.StructFilters
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.IntegerType

/**
  * Parameters of the INFFC noise filtering method
  */
trait INFFCParams extends Params
                  with HasLabelCol
                  with HasFeaturesCol
                  with Serializable {
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

/**
  * Class Noise Filtering method.
  * 
  * It combines multiple classifiers in order to remove noisy instances in
  * multiple iterations by computing a noise metric.
  * 
  * This proposal consists of three steps that are performed in each iteration:
  * 
  * - Preliminary filtering: First an ensemble of classifiers is used to filter
  *                          potential noisy examples in order to reduce its
  *                          impact in the following steps.
  * 
  * - Noise-free filtering: Another ensemble is built from the preliminary
  *                         filtered data from the previous step. Then, this
  *                         ensemble is used to predict the whole dataset.
  *                         This process results in two sets of data: a clean
  *                         set and a noisy set, both of them expected to be
  *                         more accurate since they are built from cleaner data.
  * 
  * - Final removal of noise: The final step of each iteration controls the level
  *                           of conservation of the filter.
  *                           A noise score is computed for each potentially
  *                           noisy example from the noisy set of data.
  *                           By using a configurable threshold, examples whose
  *                           noise score exceeds that threshold are filtered out.
  * 
  * This process is applied iteratively until for a configurable number
  * {@code stopCriteriaG} of consecutive iterations, the number of examples
  * tagged as noisy is less than a certain percentage {@code stopCriteriaP}
  * of the size of the original training dataset.
  */
class INFFC(override val uid: String) extends Transformer
                                      with INFFCParams
                                      with Serializable {
    private val logger = Logger.getLogger(getClass.getName)

    def this() = this(Identifiable.randomUID("INFFC"))
    
    def setLabelCol(value: String): this.type = set(labelCol, value)
    def setFeaturesCol(value: String): this.type = set(featuresCol, value)
    def setClassifiers(value: Array[Estimator[_]]): this.type = set(classifiers, value)
    def setVotingSchema(value: VotingSchema.VotingSchema): this.type = set(votingSchema, value)
    def setK(value: Int): this.type = set(K, value)
    def setNoiseScoreThreshold(value: Double): this.type = set(noiseScoreThreshold, value)
    def setStopCriteriaG(value: Int): this.type = set(stopCriteriaG, value)
    def setStopCriteriaP(value: Double): this.type = set(stopCriteriaP, value)

    /** 
     * Filters out noise from the input dataset
     * 
     * @param dataset Dataset to remove noise from
     * 
     * @return a new DataFrame with the same structure and content
     *          as {@code dataset} but with noisy rows filtered out
     */
    override def transform(dataset: Dataset[_]): DataFrame = {
        val ensemble = new VoteEnsemble()
            .setVotingSchema($(votingSchema))
            .setClassifiers($(classifiers))
            .setLabelCol($(labelCol))

        val original_df = dataset.toDF()
        var current_df = original_df

        logger.debug(s"Original size: ${original_df.count()}")

        val stopCriteriaNoiseExamples = original_df.count() * $(stopCriteriaP)
        var stopCriteriaIters = 0

        while (stopCriteriaIters < $(stopCriteriaG)) {
            logger.debug("New iteration")

            val noise_free_df = preliminaryNoiseFilter(ensemble, current_df)
            logger.debug("Done with Preliminary Noise Filtering")

            val filtered_df = noiseFreeFiltering(ensemble, current_df, noise_free_df)
            logger.debug("Done with Noise-Free Filtering")
            
            current_df.cache()
            filtered_df.cache()
            val n_noisy_examples = current_df.count() - filtered_df.count()
            logger.debug(s"$n_noisy_examples noisy examples removed")

            if (n_noisy_examples < stopCriteriaNoiseExamples) {
                stopCriteriaIters += 1
                logger.info(s"Iteration $stopCriteriaIters out of ${$(stopCriteriaG)} with samples bellow threshold")
                logger.debug(s"N. noisy: $n_noisy_examples - N. threshold: $stopCriteriaNoiseExamples")
            } else {
                stopCriteriaIters = 0
                logger.debug(s"Resetting stop iterations counter")
            }

            current_df.unpersist()
            current_df = filtered_df
        }

        logger.info("Dataset cleaned")
        logger.debug(s"Original size: ${original_df.count()} - Cleaned size: ${current_df.count()}")
        current_df
    }

    /**
      * Preliminary Noise Filtering
      * 
      * An ensemble of classifiers is used to filter
      * potential noisy examples in order to reduce its
      * impact in the following steps.
      *
      * @param vote_ensemble ensemble of classifiers for voting
      * @param dataset dataset to filter
      * @return a filtered dataset without potentially noisy examples
      */
    private[ml] def preliminaryNoiseFilter(
            vote_ensemble: VoteEnsemble,
            dataset: DataFrame): DataFrame = {    
        val vote_result = vote_ensemble.fit(dataset).transform(dataset)
        logger.debug("Trained voting ensemble on noisy set")
        vote_result.filter("noisy == false")
    }

    /**
      * Noise-free filtering
      * 
      * An ensemble is built from the preliminary
      * filtered data from the previous step. Then, this
      * ensemble is used to predict the whole dataset.
      * This process results in two sets of data: a clean
      * set and a noisy set, both of them expected to be
      * more accurate since they are built from cleaner data.
      * 
      * A noise score is computed for each potentially
      * noisy example from the noisy set of data.
      * By using a configurable threshold, examples whose
      * noise score exceeds that threshold are filtered out.
      *
      * @param vote_ensemble ensemble of classifiers for voting
      * @param dataset dataset to filter
      * @param noise_free_df dataset without potentially noisy examples
      * @return the {@code dataset} DataFrame filtered out after computing
      *         the noise score. Those examples with a noise score higher than
      *         {@code noiseScoreThreshold} are removed.
      */
    private[ml] def noiseFreeFiltering(
            vote_ensemble: VoteEnsemble,
            dataset: DataFrame,                    
            noise_free_df: DataFrame): DataFrame = {
        val vote_result = vote_ensemble
            .fit(noise_free_df)
            .transform(dataset)
            .withColumn("ID", monotonically_increasing_id())
        logger.debug("Trained voting ensemble on noise-free set")

        val knn = new KNN()
            .setTopTreeSize(10)
            .setFeaturesCol($(featuresCol))
            .setAuxCols(Array("ID", $(featuresCol), $(labelCol), "noisy"))

        // K+1 because KNN returns the sample itself as a neighbour,
        // we will discard it afterwards.
        // See: https://github.com/saurfang/spark-knn/issues/7
        val knn_model = knn.fit(vote_result).setK($(K)+1)
        val knn_result = knn_model.transform(vote_result)
        logger.debug("done with KNN")

        val df_with_noiseScore = computeNoiseScore(knn_result)
        logger.debug("NS Computed")

        // Filter clean data and drop all the columns but the original ones
        df_with_noiseScore
            .filter(s"noise_score <= ${$(noiseScoreThreshold)}")
            .select(dataset.columns(0),
                    dataset.columns.takeRight(dataset.columns.length-1): _*)
    }

    /**
      * Compute the noise score
      *
      * @param knn_df dataframe with neighbors computed
      * @return dataframe with neiughbors and examples properties and
      *         noise score computed
      */
    private[ml] def computeNoiseScore(knn_df: DataFrame): DataFrame = {
        // n(e): Number of noisy examples in CN among the k nearest neighbors
        //       of the example e
        val noisy_neighborsUDF = udf {
            (neighbors: mutable.WrappedArray[Row]) => {
                neighbors.count(_.getAs[Boolean]("noisy") == true)
            }
        }

        val confidenceUDF = udf {
            (times_in_noisy_neighbourhood: Integer) => {
                1 / Math.sqrt(1 + Math.pow(times_in_noisy_neighbourhood.toDouble, 2))
            }
        }

        val cleanUDF = udf {
            (noisy_neighbors: Integer, noisy: Boolean, k: Integer) => {
                val is_noise = if (noisy) 1 else -1;
                (k + is_noise * (noisy_neighbors - k)) / (2*k.toDouble)
            }
        }

        val neighborhoodUDF = udf {
            (label: Double, neighbors: mutable.WrappedArray[Row], k: Integer, labelColName: String) => {
                val aux_sum = neighbors.map(row => {
                        val neighbourLabel = row.getAs[Double](labelColName)

                        val clean = row.getAs[Double]("clean")
                        val confidence = row.getAs[Double]("confidence")
                        val differentClasses = if (label != neighbourLabel) 1 else -1

                        clean * confidence * differentClasses
                    }).reduce(_ + _)

                aux_sum / k.toDouble
            }
        }

        var df_with_properties = knn_df
            .withColumn("noisy_neighbors",
                        noisy_neighborsUDF(col("neighbors")))
            .withColumn("clean",
                        cleanUDF(col("noisy_neighbors"),
                                 col("noisy"),
                                 lit($(K))))
            // Make K rows for each example: one for each neighbour
            .withColumn("neighbor", explode(col("neighbors")))
                    
        // t(e): Number of times that e is among the k nearest neighbors of
        //       other noisy examples in CN
        //
        // Group by each neighbour and for each one, count the
        // number of times that noisy is true for the example
        // it is a neighbour of
        val times_in_noisy_neighbourhood = df_with_properties
            .withColumn("noisyInt", col("noisy").cast(IntegerType))
            .groupBy("neighbor.ID")
            .agg(
                sum("noisyInt").as("times_in_noisy_neighbourhood")
            )
        
        // Join the above DF so each example has its times_in_noisy_neighbourhood
        df_with_properties = df_with_properties
            .join(times_in_noisy_neighbourhood, "ID")
            .withColumn("confidence",
                        confidenceUDF(col("times_in_noisy_neighbourhood")))

        df_with_properties = df_with_properties  
            // Join so each example has also the properties fo its neighbour 
            // We need to group by the ID so we don't get duplicated records
            // (one for each neighbor exploded)
            .join(df_with_properties
                    .groupBy("ID")
                    .agg(
                        first("noisy_neighbors").as("neighbor_noisy_neighbors"),
                        first("clean").as("neighbor_clean"),
                        first("times_in_noisy_neighbourhood").as("neighbor_times_in_noisy_neighbourhood"),
                        first("confidence").as("neighbor_confidence")
                    )
                    .withColumnRenamed("ID", "neighbor_ID"),
                  col("neighbor.ID") === col("neighbor_ID"))
            // Now pack the properties we just joined into a new neighbour struct
            .withColumn("neighbor",
                        struct(
                            col("neighbor.*"),
                            col("neighbor_noisy_neighbors").as("noisy_neighbors"),
                            col("neighbor_times_in_noisy_neighbourhood").as("times_in_noisy_neighbourhood"),
                            col("neighbor_clean").as("clean"),
                            col("neighbor_confidence").as("confidence")
                        ))
            // Drop the leftovers from this operation
            .drop("neighbor_ID",
                  "neighbor_noisy_neighbors",
                  "neighbor_times_in_noisy_neighbourhood",
                  "neighbor_clean",
                  "neighbor_confidence")

        // On the original dataframe:
        knn_df
            // Drop the original neighbors and add the properties along with
            // the new neighbors with their properties collapsed into a new
            // array (i.e. undo the explode).
            .drop("neighbors")
            .join(df_with_properties
                    .drop("neighbors")
                    .groupBy(col("ID"))
                    .agg(
                        // Need to use .as(colName) since the column name
                        // will be first<colName> by default
                        first("noisy_neighbors").as("noisy_neighbors"),
                        first("times_in_noisy_neighbourhood").as("times_in_noisy_neighbourhood"),
                        first("clean").as("clean"),
                        first("confidence").as("confidence"),
                        collect_list("neighbor").as("neighbors")
                    ), "ID")
            // Finally compute the last two properties.
            .withColumn("neighborhood",
                        neighborhoodUDF(col($(labelCol)),
                                        col("neighbors"),
                                        lit($(K)),
                                        lit($(labelCol))))
            .withColumn("noise_score", 
                        col("confidence")*col("neighborhood"))
    }

    override def transformSchema(schema: StructType): StructType = schema
    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}