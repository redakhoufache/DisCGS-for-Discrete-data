package GS.Distributed
import breeze.numerics.log

import scala.collection.mutable.ArrayBuffer
import Common.Tools.{buildVocabulary, computeTotalWordsPerCluster, countWordByCluster, normalizeLogProbability, partitionToOrderedCount, sample, wordCountPerDocument}

class WorkerCGS(val workerID: Int,
                indices: Array[Int],
                data: Array[String],
                vocabulary: Array[String],
                gamma: Double,
                initByUserPartition: Option[Array[Int]] = None
               ) extends Serializable {
  val n: Int = data.length
  val alpha: Double = 0.1 * n
  val subV : Int = buildVocabulary(data).length
  val V: Int = vocabulary.length
  var globalPartition: ArrayBuffer[Int] = List.fill(n)(0).to(ArrayBuffer)
  var partition: Array[Int] = initByUserPartition match {
    case Some(m) =>
      require(m.length == data.length)
      m
    case None => Array.fill(n)(0)}

  var countCluster: ArrayBuffer[Int] = partitionToOrderedCount(partition).to(ArrayBuffer)
  var docLengths: Array[Int] = wordCountPerDocument(data)

  var wordOccurrencesByCluster: ArrayBuffer[Array[Int]] = ArrayBuffer(
    countWordByCluster(vocab = vocabulary, corpus = data, partition = partition).map(_.toArray): _*
  )
  var totalWordsPerCluster: ArrayBuffer[Int] = ArrayBuffer(
    computeTotalWordsPerCluster(countCluster, partition, docLengths): _*
  )

  val vocabularyIndex: Map[String, Int] = vocabulary.zipWithIndex.toMap

  val sparseWordCountsByDoc: Array[Array[(Int, Int)]] = data.map { doc =>
    val freqs = scala.collection.mutable.Map[Int, Int]()
    doc.split("\\s+").foreach { w =>
      vocabularyIndex.get(w).foreach { idx =>
        freqs.update(idx, freqs.getOrElse(idx, 0) + 1)
      }
    }
    freqs.toArray
  }

  def computePriorPredictive(
                              docWordCounts: Array[Array[(Int, Int)]],
                              docLengths: Array[Int],
                              V: Int,
                              gamma: Double
                            ): Array[Double] = {

    docWordCounts.indices.map { i =>
      val freqs = docWordCounts(i)
      val docLength = docLengths(i)

      val logNumerator = freqs.map { case (_, count) =>
        (0 until count).map(s => math.log(gamma + s)).sum
      }.sum

      val logDenominator = (0 until docLength).map(s => math.log(V * gamma + s)).sum

      logNumerator - logDenominator
    }.toArray
  }

  val priorPredictive: Array[Double] = computePriorPredictive(docWordCounts = sparseWordCountsByDoc, docLengths = docLengths, V = subV, gamma = gamma)

  def removeDocumentFromItsCluster(idx: Int): Unit = {
    val currentMembership = partition(idx)
    val wordCountDocument = sparseWordCountsByDoc(idx)

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
    val wordCountDocument = sparseWordCountsByDoc(idx)
    val documentLength = docLengths(idx)
    val logNumerator = wordCountDocument.map { case (wordIdx, count) =>
      val clusterWordCount = wordOccurrencesByCluster(k)(wordIdx)
      (0 until count).map(s => log(gamma + clusterWordCount + s)).sum
    }.sum

    val logDenominator = (0 until documentLength).map(s => log(subV * gamma + totalWordsPerCluster(k) + s)).sum

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
    val normalizedProbs = normalizeLogProbability(probPartition :+ probPartitionNewCluster)
    val newPartition = sample(normalizedProbs)
    partition(idx) = newPartition
  }

  def addDocumentToCluster(idx: Int): Unit = {
    val newPartition = partition(idx)
    val wordCounts = sparseWordCountsByDoc(idx)

    if (newPartition == countCluster.length) {
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


  def updateDPMWithMasterResults(results: Array[(Int, Int, Int)]): Unit = {
    require(results.length == countCluster.length, "results length is not equal to countCluster length")


    for (i <- 0 until n) {
      globalPartition.update(i, results(partition(i))._3)
    }
  }

  def getGlobalResults: List[(Int,Int)] = (indices zip globalPartition).toList

  def run(nIter: Int): (Int, ArrayBuffer[Array[Int]], ArrayBuffer[Int], ArrayBuffer[Int]) = {
    for ( iter <- 0 until nIter) {
      var drawingTime = 0.0

      for (d <- 0 until n) {
        removeDocumentFromItsCluster(d)
        val t0 = System.nanoTime()
        drawMembership(idx = d)
        val t1 = System.nanoTime()
        drawingTime += (t1-t0)/1e9
        addDocumentToCluster(d)
      }
      println(s"worker ID ${workerID}, worker iter ${iter} ,drawing time ${drawingTime}, local clusters ${countCluster.length}")
    }

    (workerID, wordOccurrencesByCluster, totalWordsPerCluster, countCluster)
  }
}