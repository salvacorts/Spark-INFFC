package org.apache.spark.ml.noise

import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructField
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.sql.types.BooleanType
import org.apache.log4j.Logger
import org.apache.log4j.Level

object VotingSchema extends Enumeration{
    type VotingSchema = Value
    val Consensus, Majority = Value
}

trait VoteEnsembleParams extends Params with HasLabelCol {
    val votingSchema = new Param[VotingSchema.VotingSchema](this, "votingSchema", "Voting schema")
    def getVotingSchema: VotingSchema.VotingSchema = $(votingSchema)
    setDefault(votingSchema, VotingSchema.Consensus)

    val classifiers = new Param[Array[Estimator[_]]](this, "classifiers", "Array of classifiers to use")
    def getClassifiers: Array[Estimator[_]] = $(classifiers)
}

class VoteEnsembleModel(override val uid: String,
                        val models: Array[Model[_]])
        extends Model[VoteEnsembleModel] with VoteEnsembleParams {

    def this(models: Array[Model[_]]) =
        this(Identifiable.randomUID("VoteEnsembleModel"), models)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val dataset_ids = dataset.withColumn("id", monotonicallyIncreasingId)

        val predictions = models.zipWithIndex.map{
            case(model, idx) => {
                val pred_df = model.transform(dataset_ids)
                    .select("id", "prediction")
                    .withColumnRenamed("prediction", s"prediction_$idx")

                (pred_df, s"prediction_$idx")
            }
        }

        val pred_dfs = predictions.map(_._1)
        val pred_cols = predictions.map(_._2)

        val merged_predicitons = pred_dfs.reduce(_.join(_, "id"))
        val dataset_with_votes = dataset_ids.join(merged_predicitons, "id")
            .withColumn("predictions", array(pred_cols.map(col(_)):_*))
            .drop(pred_cols:_*)
            .drop("id")

        val noisyUDF = udf { (predictions: Seq[Double], label: Double) => {
            val votes = predictions.map(p => {if (p != label) true else false})
            if ($(votingSchema) == VotingSchema.Majority) {
                votes.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
            } else { // Consensus
                votes.forall(_==true)
            }
        }}

        dataset_with_votes.withColumn(
            "noisy", noisyUDF(col("predictions"), col($(labelCol))))
    }

    override def transformSchema(schema: StructType): StructType = {
        StructType(schema.fields ++ Array(
            new StructField("predictions", ArrayType(DoubleType), true),
            new StructField("noisy", BooleanType, true)))
    }

    override def copy(extra: ParamMap): VoteEnsembleModel = defaultCopy(extra)
}

class VoteEnsemble(override val uid: String) extends Estimator[VoteEnsembleModel]
                                             with VoteEnsembleParams {
    private val logger = Logger.getLogger(getClass.getName)

    def this() = this(Identifiable.randomUID("VoteEnsemble"))

    def setClassifiers(value: Array[Estimator[_]]): this.type = set(classifiers, value)
    def setVotingSchema(value: VotingSchema.VotingSchema): this.type = set(votingSchema, value)
    def setLabelCol(value: String): this.type =  set(labelCol, value)

    override def fit(dataset: Dataset[_]): VoteEnsembleModel = {  
        if (logger.getLevel() == Level.DEBUG) {
            val str = new StringBuilder("Using these classifiers:\n")
            $(classifiers).foreach(c => {str ++= s"\t${c.getClass.getName}\n"})
            logger.debug(str)
        }
        
        val trained_models = $(classifiers).map(classifier => {
            classifier.fit(dataset).asInstanceOf[Model[_]]
        })

        copyValues(new VoteEnsembleModel(trained_models).setParent(this))
    }
  
    override def transformSchema(schema: StructType): StructType = schema
    override def copy(extra: ParamMap): Estimator[VoteEnsembleModel] = defaultCopy(extra)
}
