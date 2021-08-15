package org.apache.spark.ml.noise

import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.{Params, Param, ParamMap}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, DataFrame}
import scala.util.Random

trait RandomNoiseParams extends Params with HasLabelCol {
    val noisePercentage = new Param[Double](this, "noisePercentage", "Percentage of examples to add noise to")
    def getNoisePercentage: Double = $(noisePercentage)
    setDefault(noisePercentage, 0.2)
}

class RandomNoise(override val uid: String) extends Transformer
                                            with RandomNoiseParams {
    def this() = this(Identifiable.randomUID("RandomNoise"))

    def setLabelCol(value: String): this.type = set(labelCol, value)
    def setNoisePercentage(value: Double): this.type = set(noisePercentage, value)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val sc = dataset.sparkSession.sparkContext
    
        val labels = dataset
            .select(col($(labelCol)))
            .distinct()
            .collect()
            .map(_.getAs[Double]($(labelCol)))
        val labels_broadcast = sc.broadcast(labels)

        val df_idx = dataset.withColumn("RandomNoiseIDX", monotonicallyIncreasingId)

        val indexes = df_idx
            .select("RandomNoiseIDX")
            .collect()
            .map(_.getAs[Long]("RandomNoiseIDX"))
            .toSeq

        val df_size = dataset.count().toInt
        val n_examples_noise = math.round(df_size * $(noisePercentage)).toInt

        val noise_indexes = Random.shuffle(indexes).take(n_examples_noise)
        val noise_indexes_broadcast = sc.broadcast(noise_indexes)

        val noiseUDF = udf {
            (idx: Long, original_label: Double) => {
                if (noise_indexes_broadcast.value.contains(idx)) {
                    val candidate_labels = labels_broadcast.value.diff(List(original_label))
                    candidate_labels(Random.nextInt(candidate_labels.length))
                } else {
                    original_label
                }
            }
        }

        df_idx.withColumn($(labelCol),
                          noiseUDF(col("RandomNoiseIDX"),
                                   col($(labelCol))))
            .drop("RandomNoiseIDX")
    }

    override def transformSchema(schema: StructType): StructType = schema
    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}