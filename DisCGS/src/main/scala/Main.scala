import Common.ExperimentResultFinal
import GS.{CGS, DisCGS}
import org.json4s.DefaultFormats
import Common.Tools.{buildVocabulary, createSparkContext, loadDataset, mean, saveJson, stddev}
import org.apache.spark.{SparkConf, SparkContext}
import smile.validation.{NormalizedMutualInformation, adjustedRandIndex}

import scala.collection.mutable.ListBuffer

object Main {

  implicit val formats: org.json4s.Formats = DefaultFormats

  def main(args: Array[String]): Unit = {

    val datasetName = args(0)
    val gamma = args(1).toDouble
    val nIterations = args(2).toInt
    val distributed = args(3).toBoolean
    val nbWorkers = args(4).toInt

    val (corpus, trueLabels) = loadDataset(datasetName)
    val vocabulary = buildVocabulary(corpus)

    val trueK = trueLabels.distinct.length

      val scOpt = if (distributed) Some(createSparkContext(nbWorkers, "DPMM Experiment1")) else None

     if (distributed){
       println(s"Running DisCGS on ${datasetName} dataset using ${nbWorkers} workers for ${nIterations} iterations...")
     }else{
       println(s"Running CGS on ${datasetName} dataset for ${nIterations} iterations...")
     }

      val (finalPartition, runningTime) = if (distributed) {
        val sc = scOpt.get
        val rdd = sc.parallelize(corpus.zipWithIndex.map(_.swap), nbWorkers)
        val model = new DisCGS(rdd, vocabulary, masterGamma = gamma * nbWorkers / 2, workerGamma = gamma, n = corpus.length, nbWorkers = nbWorkers)
        val start = System.nanoTime();
        model.run(nIterations);
        val end = System.nanoTime()
        (model.partitionEveryIteration.last, (end - start) / 1e9)
      } else {
        val model = new CGS(data = corpus, vocabulary = vocabulary, alpha = 0.01 * corpus.length, gamma = gamma)
        val start = System.nanoTime();
        model.run(nIterations);
        val end = System.nanoTime()
        (model.partitionEveryIteration.last, (end - start) / 1e9)
      }

      if(distributed){scOpt.get.stop()}
      val ari = adjustedRandIndex(finalPartition, trueLabels)
      val nmi = NormalizedMutualInformation.sum(finalPartition, trueLabels)
      val inferredK = finalPartition.distinct.length

    val results = Map(
      "Gamma" -> gamma,
      "ARI" -> ari,
      "NMI" -> nmi,
      "Running time" -> runningTime,
      "True clusters" -> trueK,
      "Inferred clusters" -> inferredK
    )

    println("Results:")
    results.foreach { case (key, value) => println(f"$key%-18s: $value") }

  }
}
