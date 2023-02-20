import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{IntegerType, DoubleType}
import org.apache.spark.sql.DataFrame

val schema_covid19 = StructType( StructField("_id", IntegerType, nullable = true) :: 
                              StructField("Assigned_ID", IntegerType, nullable = true) :: 
                              StructField("Outbreak Associated", StringType, nullable = true) ::
                              StructField("Age Group", StringType, nullable = true) ::
                              StructField("Neighbourhoood Name", StringType, nullable = true) :: 
                              StructField("FSA", StringType, nullable = true) :: 
                              StructField("Source of Infection", StringType, nullable = true) :: 
                              StructField("Classification", StringType, nullable = true) :: 
                              StructField("Episode Date", DateType, nullable = true) :: 
                              StructField("Reported Date", DateType, nullable = true) :: 
                              StructField("Client Gender", StringType, nullable = true) :: 
                              StructField("Outcome", StringType, nullable = true) :: 
                              StructField("Currently Hospitalized", StringType, nullable = true) :: 
                              StructField("Currently in ICU", StringType, nullable = true) :: 
                              StructField("Currently Intubated", StringType, nullable = true) :: 
                              StructField("Ever Hospitalized", StringType, nullable = true) :: 
                              StructField("Ever in ICU", StringType, nullable = true) :: 
                              StructField("Ever Intubated", StringType, nullable = true) :: Nil )

val raw_covid19_df = spark.read.format("csv").
                                option("header", value = true).option("delimiter", ",").option("mode", "DROPMALFORMED").
                                schema(schema_covid19).load("hdfs:/BigData/Covid19Cases.csv").cache()

raw_covid19_df.printSchema()

raw_covid19_df.show(2)

for( i <- raw_covid19_df.columns){
    println(i+" "+raw_covid19_df.filter(raw_covid19_df(i).isNull || raw_covid19_df(i) === "" ).count()+" "+ 
    100*(raw_covid19_df.filter(raw_covid19_df(i).isNull || raw_covid19_df(i) === "" ).count())/(raw_covid19_df.count()))
}

val covid19_df = raw_covid19_df.filter(col("Outcome").isin("RESOLVED","FATAL"))
                .filter(col("Age Group").isNotNull)

covid19_df.count()

val indexer = new StringIndexer()
  .setInputCol("Outcome")
  .setOutputCol("OutcomeIDX") 

print(indexer)

val covid19_df1 = indexer.fit(covid19_df).transform(covid19_df)

/*import scala.collection.mutable.ListBuffer*/
var f_indexers = new Array[org.apache.spark.ml.PipelineStage](0)
val featuresList = List("Outbreak Associated","Age Group","Source of Infection","Client Gender","Ever Hospitalized",
                    "Ever in ICU","Ever Intubated")

for (feature <- featuresList){
    print(feature)
    val f_indexer = new StringIndexer().setInputCol(feature).setOutputCol(feature+ " IDX")
    print(f_indexer)
    f_indexers = f_indexers :+ f_indexer
}
f_indexers

val fpipeline =  new Pipeline()
  .setStages(f_indexers)

val covid19_df2= fpipeline.fit(covid19_df1).transform(covid19_df1)

covid19_df2.filter(col("Outcome")==="FATAL").show(2)

val resolved_df = covid19_df2.filter(col("Outcome") === "RESOLVED")
val fatal_df = covid19_df2.filter(col("Outcome") === "FATAL")
val ratio:Double =  (resolved_df.count()/fatal_df.count())

// Under Sampling 'resolved' records and combining with 'fatal' population in order to generate a balanced dataset
val r:Double = (1/ratio).toDouble
print(r)
val sampled_resolved_df = resolved_df.sample(false,r, 42)
val combined_sampled_df = sampled_resolved_df.unionAll(fatal_df)
combined_sampled_df.filter(col("Outcome")==="FATAL").count()

val assembler = new VectorAssembler()
 .setInputCols(Array("Outbreak Associated IDX","Age Group IDX","Source of Infection IDX","Client Gender IDX","Ever Hospitalized IDX",
                    "Ever in ICU IDX","Ever Intubated IDX"))
 .setOutputCol("assembled-features")

val rf = new RandomForestClassifier()
 .setFeaturesCol("assembled-features")
 .setLabelCol("OutcomeIDX")
 .setSeed(42)

val pipeline = new Pipeline()
  .setStages(Array(assembler, rf))

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("OutcomeIDX")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val paramGrid = new ParamGridBuilder()  
  .addGrid(rf.maxDepth, Array(3, 4))
  .addGrid(rf.impurity, Array("entropy","gini")).build()

val cross_validator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

/* val trainData = covid19_weighted.sample("OutcomeIDX", fractions={0.0: 0.09, 1.0: 0.7}, seed=42)*/

val Array(trainingData, testData) = combined_sampled_df.randomSplit(Array(0.8, 0.2), 40)

val cvModel = cross_validator.fit(trainingData)

val predictions = cvModel.transform(testData)

val accuracy = evaluator.evaluate(predictions)
print(accuracy)
