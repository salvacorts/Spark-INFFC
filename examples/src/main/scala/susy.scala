import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.VectorAssembler

object SUSY {
    def main(args: Array[String]) = {
        val spark = SparkSession
            .builder()
            .master(sys.env("SPARK_MASTER"))
            .appName("Spark-INFCC SUSY")
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        val susy_train_path = args(0)
        val susy_test_path = args(1)
        val output_path = args(2)
        val noise_percentage = args(3).toDouble

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

        val clean_train_df = Benchmark.runINFFC(noise_percentage,
                                                train_df_ml,
                                                test_df_ml)
        
        spark.close()
    }
}