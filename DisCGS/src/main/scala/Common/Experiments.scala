package Common

import Common.Tools.{buildVocabulary, createSparkContext, loadDataset, mean, stddev}
import GS.{CGS, DisCGS}
import org.apache.spark.{SparkConf, SparkContext}
import smile.validation.{NormalizedMutualInformation, adjustedRandIndex}

import scala.collection.mutable.ListBuffer

case class ExperimentResultFinal(
                                  experiment: Int,
                                  dataset: String,
                                  params: Map[String, Any],
                                  results: Map[String, Any]
                                )

class Experiments {

  def experiment1(datasetNames: Seq[String], nIterations: Int, nRuns: Int,  gammaValues: Seq[Double], distributed: Boolean, nbWorkersValues: Seq[Int]): ExperimentResultFinal = {

    val avgARIs: ListBuffer[Double] = ListBuffer()
    val sdARIs: ListBuffer[Double] = ListBuffer()
    val Gammas: ListBuffer[Double] = ListBuffer()
    val avgNMIs: ListBuffer[Double] = ListBuffer()
    val sdNMIs: ListBuffer[Double] = ListBuffer()
    val avgInferredKs: ListBuffer[Double] = ListBuffer()
    val avgRunningTimes: ListBuffer[Double] = ListBuffer()
    val sdRunningTimes: ListBuffer[Double] = ListBuffer()
    val TruesKs: ListBuffer[Int] = ListBuffer()

    for (idx <- datasetNames.indices){

      val datasetName = datasetNames(idx)
      val gamma = gammaValues(idx)
      val nbWorkers = nbWorkersValues(idx)

      println(s"Running Experiment 2 : Clustering quality and running time on dataset $datasetName")
      val (corpus, trueLabels) = loadDataset(datasetName)
      val vocabulary = buildVocabulary(corpus)
      val trueK = trueLabels.distinct.length

      val scOpt = if (distributed) Some(createSparkContext(nbWorkers, "DPMM Experiment1")) else None

      val results = (0 until nRuns).map { run =>
        val (finalPartition, runningTime) = if (distributed) {
          val sc = scOpt.get
          val rdd = sc.parallelize(corpus.zipWithIndex.map(_.swap), nbWorkers)
          val model = new DisCGS(rdd, vocabulary, masterGamma = gamma * nbWorkers, workerGamma = gamma, n = corpus.length, nbWorkers = nbWorkers)
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

        (adjustedRandIndex(finalPartition, trueLabels),
          NormalizedMutualInformation.sum(finalPartition, trueLabels),
          finalPartition.distinct.length,
          runningTime)
      }.toArray

      scOpt.get.stop()

      avgARIs += mean(results.map(_._1))
      sdARIs += stddev(results.map(_._1))
      Gammas += gamma
      avgNMIs += mean(results.map(_._2))
      sdNMIs += stddev(results.map(_._2))
      avgInferredKs += mean(results.map(_._3).map(_.toDouble))
      avgRunningTimes += mean(results.map(_._4))
      sdRunningTimes += stddev(results.map(_._4))
      TruesKs += trueK
    }

    ExperimentResultFinal(
      experiment = 1,
      dataset = "None",
      params = Map(
        "datasetNames" -> datasetNames,
        "Gammas" -> Gammas,
        "distributed" -> distributed,
        "nbWorkers" -> nbWorkersValues
      ),
      results = Map(
        "bestARIs" -> avgARIs,
        "sdARIs" -> sdARIs,
        "Gammas"  -> Gammas,
        "avgNMIs" -> avgNMIs,
        "sdNMIs" -> sdNMIs,
        "avgInferredKs" -> avgInferredKs,
        "avgRunningTimes" -> avgRunningTimes,
        "sdRunningTimes" -> sdRunningTimes,
        "TruesKs" -> TruesKs
      )
    )
  }


  def experiment2(datasetName: String, nIterations: Int,  gammaValues: Seq[Double], datasetSizes: Seq[Int], distributed: Boolean, nbWorkersValues: Seq[Int]): ExperimentResultFinal = {

    println(s"Running Experiment 2 : Clustering quality and running time on dataset $datasetName")
    val (allCorpus, allTrueLabels) = loadDataset(datasetName)

    val ARIs: ListBuffer[Double] = ListBuffer()
    val Gammas: ListBuffer[Double] = ListBuffer()
    val NMIs: ListBuffer[Double] = ListBuffer()
    val InferredKs: ListBuffer[Int] = ListBuffer()
    val RunningTimes: ListBuffer[Double] = ListBuffer()
    val TruesKs: ListBuffer[Int] = ListBuffer()

    for (sizeIdx <- datasetSizes.indices) {
      val gamma = gammaValues(sizeIdx)
      val dataSetSize = datasetSizes(sizeIdx)
      val nbWorkers = nbWorkersValues(sizeIdx)
      val corpus = allCorpus.take(dataSetSize)
      val trueLabels = allTrueLabels.take(dataSetSize)
      val vocabulary = buildVocabulary(corpus)
      val trueK = trueLabels.distinct.length

      val scOpt = if (distributed) Some(createSparkContext(nbWorkers, "DPMM Experiment1")) else None

      println(f"Running on a dataset of size: ${dataSetSize}")
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

      scOpt.get.stop()
      ARIs += adjustedRandIndex(finalPartition, trueLabels)
      Gammas += gamma
      NMIs += NormalizedMutualInformation.sum(finalPartition, trueLabels)
      InferredKs += finalPartition.distinct.length
      RunningTimes += runningTime
      TruesKs += trueK

    }

    ExperimentResultFinal(
      experiment = 2,
      dataset = datasetName,
      params = Map(
        "Gammas" -> gammaValues,
        "distributed" -> distributed,
        "nbWorkers" -> nbWorkersValues
      ),
      results = Map(
        "ARIs" -> ARIs,
        "NMIs" -> NMIs,
        "RunningTimes" -> RunningTimes,
        "trueClusters" ->TruesKs,
        "inferredClustersValues" -> InferredKs
      )
    )
  }

  def experiment3(datasetName: String, nIterations: Int,  gammaValues: Seq[Double], nbWorkersValues: Seq[Int]): ExperimentResultFinal = {
    println(s"Running Experiment 3: Scale up")

    val (corpus, trueLabels) = loadDataset(datasetName)
    val vocabulary = buildVocabulary(corpus)
    val trueK = trueLabels.distinct.length
    val ARIs: ListBuffer[Double] = ListBuffer()
    val gammas: ListBuffer[Double] = ListBuffer()
    val NMIs: ListBuffer[Double] = ListBuffer()
    val InferredKs: ListBuffer[Int] = ListBuffer()
    val runningTimes: ListBuffer[Double] = ListBuffer()

    for (nbworkerIdx <- nbWorkersValues.indices){
      val workers = nbWorkersValues(nbworkerIdx)
      val gamma = gammaValues(nbworkerIdx)
      val conf = new SparkConf()
        .setMaster(s"local[$workers]")
        .setAppName(s"DisCGS-${workers}Workers")
        .set("spark.driver.memory",       "128g")
        .set("spark.driver.maxResultSize","96g")
        .set("spark.executor.memory",     "32g")
        .set("spark.sql.shuffle.partitions",(workers * 6).toString)
        .set("spark.default.parallelism",  (workers * 6).toString)
        .set("spark.memory.fraction",        "0.8")
        .set("spark.memory.storageFraction", "0.3")

      val sc = new SparkContext(conf)

      val (finalPartition, runningTime) = {
        val rdd = sc.parallelize(corpus.zipWithIndex.map(_.swap), workers)
        val model = new DisCGS(rdd, vocabulary, masterGamma = gamma*workers/2, workerGamma = gamma, n=corpus.length, nbWorkers = workers)
        val start = System.nanoTime(); model.run(nIterations); val end = System.nanoTime()
        (model.partitionEveryIteration.last, (end - start) / 1e9)
      }

      sc.stop()

      ARIs +=  adjustedRandIndex(finalPartition, trueLabels)
      gammas += gamma
      NMIs += NormalizedMutualInformation.sum(finalPartition, trueLabels)
      InferredKs += finalPartition.distinct.length
      runningTimes += runningTime

    }

    ExperimentResultFinal(
      experiment = 3,
      dataset = datasetName,
      params = Map(
        "bestGammas" -> gammas,
        "distributed" -> "true",
        "nbWorkers" -> nbWorkersValues
      ),
      results = Map(
        "bestARIs" -> ARIs,
        "bestNMIs" -> NMIs,
        "bestRunningTimes" -> runningTimes,
        "trueClusters" ->"20",
        "inferredClustersValues" -> InferredKs
      )
    )
  }

}
