package GS
import breeze.numerics.log
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import Common.Tools.{computeTotalWordsPerCluster, countWordByCluster, normalizeLogProbability, partitionToOrderedCount, sample, wordCountPerDocument}

class CGS(data: Array[String],
          vocabulary: Array[String],
          alpha: Double,
          gamma: Double,
          computeLikelihood: Boolean = false) extends Serializable {

  val n: Int = data.length
  val V: Int = vocabulary.length
  var partition: Array[Int] = Array.fill(n)(0)
  var partitionEveryIteration: ArrayBuffer[Array[Int]] = ArrayBuffer(partition)
  var likelihoodEveryIteration: ArrayBuffer[Double] = ArrayBuffer()
  var docLengths: Array[Int] = wordCountPerDocument(data)
  var countCluster: ArrayBuffer[Int] = ArrayBuffer(partitionToOrderedCount(partition): _*)
  var wordOccurrencesByCluster: ArrayBuffer[Array[Int]] = ArrayBuffer(
    countWordByCluster(vocab = vocabulary, corpus = data, partition = partition).map(_.toArray): _*
  )
  var totalWordsPerCluster: ArrayBuffer[Int] = ArrayBuffer(
    computeTotalWordsPerCluster(countCluster, partition, docLengths): _*
  )

  val vocabularyIndex: Map[String, Int] = vocabulary.zipWithIndex.toMap
  val docWordCounts: Array[Array[(Int, Int)]] = data.map { doc =>
    val freqs = scala.collection.mutable.Map[Int, Int]()
    doc.split("\\s+").foreach { w =>
      vocabularyIndex.get(w).foreach { idx =>
        freqs.update(idx, freqs.getOrElse(idx, 0) + 1)
      }
    }
    freqs.toArray
  }

  def computePriorPredictive(): Array[Double] = {
    data.par.map { doc =>
      val tokens = doc.split("\\s+")
      val freqMap = tokens.groupMapReduce(identity)(_ => 1)(_ + _)
      val docLength = tokens.length
      val logNumerator = freqMap.values.map { count =>
        (0 until count).map(s => math.log(gamma + s)).sum
      }.sum

      val logDenominator = (0 until docLength).map(s => math.log(V * gamma + s)).sum

      logNumerator - logDenominator
    }.toArray
  }
  val priorPredictive: Array[Double] = computePriorPredictive()

  def removeDocumentFromItsCluster(idx: Int): Unit = {
    val currentMembership = partition(idx)
    val wordCountDocument = docWordCounts(idx)

    if (countCluster(currentMembership) == 1) {
      countCluster.remove(currentMembership)
      partition = partition.map(c => if (c > currentMembership) c - 1 else c)
      totalWordsPerCluster.remove(currentMembership)
      wordOccurrencesByCluster.remove(currentMembership)
    } else {
      countCluster.update(currentMembership, countCluster(currentMembership) - 1)

      val clusterWordCounts = wordOccurrencesByCluster(currentMembership)

      wordCountDocument.foreach { case (wordIdx, count) =>
        clusterWordCounts(wordIdx) -= count
      }
      wordOccurrencesByCluster.update(currentMembership, clusterWordCounts)
      totalWordsPerCluster.update(currentMembership, totalWordsPerCluster(currentMembership) - docLengths(idx))
    }
  }

  def posteriorPredictive(idx: Int, k: Int): Double = {
    val wordCountDocument = docWordCounts(idx)
    val documentLength = wordCountDocument.map(_._2).sum

    val logNumerator = wordCountDocument.map { case (wordIdx, count) =>
      val clusterWordCount = wordOccurrencesByCluster(k)(wordIdx)
      (0 until count).map(s => log(gamma + clusterWordCount + s)).sum
    }.sum

    val logDenominator = (0 until documentLength).map(s => log(V * gamma + totalWordsPerCluster(k) + s)).sum

    logNumerator - logDenominator
  }

  def computeClusterPartitionProbabilities(idx: Int): Array[Double] = {
    val clusterLogProbs = countCluster.indices.map { k =>
      log(countCluster(k).toDouble) + posteriorPredictive(idx, k)
    }.toArray
    clusterLogProbs
  }

  def drawMembership(idx: Int): Unit = {
    val probPartition = computeClusterPartitionProbabilities(idx)
    val probPartitionNewCluster = log(alpha) + priorPredictive(idx)
    val normalizedProbs = normalizeLogProbability((probPartition.toList :+ probPartitionNewCluster).toArray)
    val newPartition = sample(normalizedProbs)
    partition(idx) = newPartition
  }

  def addDocumentToCluster(idx: Int): Unit = {
    val newPartition = partition(idx)
    val wordCounts = docWordCounts(idx)

    if (newPartition == countCluster.length) { // New cluster
      countCluster += 1
      totalWordsPerCluster += docLengths(idx)

      val newWordCounts = Array.fill(V)(0)
      wordCounts.foreach { case (wordIdx, count) =>
        newWordCounts(wordIdx) = count
      }
      wordOccurrencesByCluster += newWordCounts

    } else {
      countCluster(newPartition) += 1
      totalWordsPerCluster(newPartition) += docLengths(idx)

      val clusterWordCounts = wordOccurrencesByCluster(newPartition)
      wordCounts.foreach { case (wordIdx, count) =>
        clusterWordCounts(wordIdx) += count
      }
      wordOccurrencesByCluster.update(newPartition, clusterWordCounts)

    }
  }

  def run(nIter: Int): Unit = {
    for (iter <- 0 until nIter) {
      var drawingTime = 0.0
      println(s"Iteration $iter")
      for (d <- 0 until n) {
        removeDocumentFromItsCluster(d)
        var t0 = System.nanoTime()
        drawMembership(d)
        var t1 = System.nanoTime()
        drawingTime +=  (t1-t0)/1e9
        addDocumentToCluster(d)
      }
      println("runTime", drawingTime)

      partitionEveryIteration += partition.clone()
      if (computeLikelihood) {
      }
    }
  }
}