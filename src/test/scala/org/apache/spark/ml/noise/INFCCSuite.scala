package org.apache.spark.ml.noise

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.BooleanType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import scala.collection.mutable
import util.Random
import org.apache.log4j.{Logger, Level}

class INFCCSuite
        extends AnyFunSuite
        with Matchers
        with BeforeAndAfterAll  {

    @transient var spark: SparkSession = null

    override def beforeAll(): Unit = {
        spark = SparkSession
            .builder()
            .master("local[*]")
            .appName("Spark-INFCC tests")
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
    }

    override def afterAll(): Unit = {
        spark.close()
    }

    test("Noise Score") {
        val data = Seq(
            Row(1, 1234, 0.0, Array(Row(2, 2234, 1.0, false), Row(3, 3234, 0.0, false)), true),
            Row(2, 2234, 1.0, Array(Row(1, 1234, 0.0, true), Row(5, 5234, 1.0, true)), false),
            Row(3, 3234, 0.0, Array(Row(4, 4234, 1.0, true), Row(5, 5234, 1.0, true)), false),
            Row(4, 4234, 1.0, Array(Row(1, 1234, 0.0, true), Row(2, 2234, 1.0, false)), true),
            Row(5, 5234, 1.0, Array(Row(2, 2234, 1.0, false), Row(4, 4234, 1.0, true)), true)
        )

        val schema = new StructType()
            .add("ID", IntegerType)
            .add("features", IntegerType)
            .add("label", DoubleType)
            .add("neighbors", ArrayType(new StructType()
                .add("ID", IntegerType)
                .add("features", IntegerType)
                .add("label", DoubleType)
                .add("noisy", BooleanType)
            ))
            .add("noisy", BooleanType)

        val df = spark.createDataFrame(
            spark.sparkContext.parallelize(data),
            schema)

        val infcc = new INFFC().setK(2)

        val df_ns = infcc.computeNoiseScore(df)
        df_ns.cache()

        df_ns.printSchema()
        df_ns.show()

        val expected_ns = Seq(
            Row(1, -0.06909830056250524),
            Row(2, -0.03952847075210474),
            Row(3, 0.15088834764831843),
            Row(4, -0.05590169943749474),
            Row(5, -0.16744528915252793)
        )
    
        val expected_ns_schema = new StructType().
            add("ID", IntegerType).
            add("expected_noise_score", DoubleType)

        val expected_ns_df = spark.createDataFrame(
            spark.sparkContext.parallelize(expected_ns),
            expected_ns_schema)
        expected_ns_df.cache()

        df_ns.count() shouldBe df.count()

        df_ns.join(expected_ns_df, "ID").collect.foreach(row => {
            val noise_score = row.getAs[Double]("noise_score")
            val expected_noise_score = row.getAs[Double]("expected_noise_score")
            
            noise_score shouldBe expected_noise_score
        })
    }

    test("Random Noise") {
        val data = Seq(
            Row(0, 0.0),
            Row(1, 1.0),
            Row(2, 2.0),
            Row(3, 0.0),
            Row(4, 1.0),
            Row(5, 2.0),
            Row(6, 0.0),
            Row(7, 1.0),
            Row(8, 2.0),
            Row(9, 0.0),
        )

        val schema = new StructType()
            .add("ID", IntegerType)
            .add("label", DoubleType)

        val df = spark.createDataFrame(
            spark.sparkContext.parallelize(data),
            schema)

        val noise_percentage = 0.5
        val random_noise = new RandomNoise()
            .setLabelCol("label")
            .setNoisePercentage(noise_percentage)

        val noisy_df = random_noise.transform(df)

        val noisy_samples = noisy_df
            .withColumnRenamed("label", "noisyLabel")
            .join(df, "ID")
            .filter(r => {
                r.getAs[Double]("label") != r.getAs[Double]("noisyLabel")
            })

        noisy_samples.count() shouldBe math.round(df.count() * noise_percentage)
    }
}