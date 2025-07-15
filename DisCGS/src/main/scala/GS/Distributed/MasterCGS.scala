package GS.Distributed
import breeze.numerics.log
import scala.collection.mutable.ArrayBuffer
import Common.Tools.{normalizeLogProbability, sample}
import scala.collection.parallel.CollectionConverters._

class MasterCGS( workerResults: Array[(Int, Int, Array[Int], Int, Int)],
                 V: Int,
                 gamma: Double,
                 nbWorkers: Int
               ) extends Serializable {

  val n: Int = workerResults.length
  val alpha: Double = n * 0.1 / nbWorkers

  // Extract data from workers
  val clusterSizes: Array[Int] = workerResults.map(_._5)
  val totalWordsByCluster: Array[Int] = workerResults.map(_._4)
  val wordOccurrencesByCluster: Array[Array[Int]] = workerResults.map(_._3)

  // Precompute and store non-zero word indices per cluster to avoid looping over all V
  val nonZeroWordIndicesByCluster: Array[Array[Int]] = wordOccurrencesByCluster.map { counts =>
    counts.zipWithIndex.collect { case (c, idx) if c > 0 => idx }
  }

  var partition: Array[Int] = Array.fill(n)(0)
  var countGlobalCluster: ArrayBuffer[Int] = ArrayBuffer(clusterSizes.sum)
  var totalWordsByGlobalCluster: ArrayBuffer[Int] = ArrayBuffer(totalWordsByCluster.sum)
  var wordOccurrencesByGlobalCluster: ArrayBuffer[Array[Int]] = ArrayBuffer(Array.ofDim[Int](V))

  // Aggregate initial global word counts
  for (localCounts <- wordOccurrencesByCluster) {
    var i = 0
    while (i < V) {
      wordOccurrencesByGlobalCluster(0)(i) += localCounts(i)
      i += 1
    }
  }

  val priorPredictive: Array[Double] = {
    val result = new Array[Double](n)
    (0 until n).par.foreach { h =>
      val wordCounts = wordOccurrencesByCluster(h)
      val nonZeroIndices = nonZeroWordIndicesByCluster(h)
      val totalWords = totalWordsByCluster(h)

      var logNumerator = 0.0
      var j = 0
      while (j < nonZeroIndices.length) {
        val w = nonZeroIndices(j)
        val count = wordCounts(w)
        var s = 0
        while (s < count) {
          logNumerator += math.log(gamma + s)
          s += 1
        }
        j += 1
      }

      var logDenominator = 0.0
      var s = 0
      while (s < totalWords) {
        logDenominator += math.log(V * gamma + s)
        s += 1
      }

      result(h) = logNumerator - logDenominator
    }
    result
  }

  def removeLocalClusterFromItsGlobalCluster(idx: Int): Unit = {
    val globalIdx = partition(idx)
    val localCounts = wordOccurrencesByCluster(idx)
    val localTotalWords = totalWordsByCluster(idx)

    if (countGlobalCluster(globalIdx) == 1) {
      countGlobalCluster.remove(globalIdx)
      totalWordsByGlobalCluster.remove(globalIdx)
      wordOccurrencesByGlobalCluster.remove(globalIdx)

      // Update partitions for removed cluster
      var i = 0
      while (i < n) {
        if (partition(i) > globalIdx) partition(i) -= 1
        i += 1
      }

    } else {
      countGlobalCluster(globalIdx) -= clusterSizes(idx)
      totalWordsByGlobalCluster(globalIdx) -= localTotalWords

      val globalCounts = wordOccurrencesByGlobalCluster(globalIdx)
      // Iterate only over non-zero word indices in the local cluster counts
      val nonZeroIndices = localCounts.zipWithIndex.collect { case (count, w) if count > 0 => w }

      var i = 0
      while (i < nonZeroIndices.length) {
        val w = nonZeroIndices(i)
        globalCounts(w) -= localCounts(w)
        i += 1
      }
      wordOccurrencesByGlobalCluster.update(globalIdx, globalCounts)
    }
  }

  def addLocalClusterToGlobalCluster(idx: Int): Unit = {
    val globalIdx = partition(idx)
    val localCounts = wordOccurrencesByCluster(idx)
    val localTotalWords = totalWordsByCluster(idx)

    if (globalIdx == countGlobalCluster.length) {
      countGlobalCluster += clusterSizes(idx)
      totalWordsByGlobalCluster += localTotalWords
      wordOccurrencesByGlobalCluster += localCounts.clone()
    } else {
      countGlobalCluster(globalIdx) += clusterSizes(idx)
      totalWordsByGlobalCluster(globalIdx) += localTotalWords

      val globalCounts = wordOccurrencesByGlobalCluster(globalIdx)
      val nonZeroIndices = localCounts.zipWithIndex.collect { case (count, w) if count > 0 => w }

      var i = 0
      while (i < nonZeroIndices.length) {
        val w = nonZeroIndices(i)
        globalCounts(w) += localCounts(w)
        i += 1
      }
      wordOccurrencesByGlobalCluster.update(globalIdx, globalCounts)
    }
  }

  def logPosteriorPredictive(idx: Int, k: Int): Double = {
    val localCounts = wordOccurrencesByCluster(idx)
    val globalCounts = wordOccurrencesByGlobalCluster(k)
    val nonZeroIndices = nonZeroWordIndicesByCluster(idx)

    val totalLocalWords = totalWordsByCluster(idx)
    val totalGlobalWords = totalWordsByGlobalCluster(k)

    var logNumerator = 0.0
    var j = 0
    while (j < nonZeroIndices.length) {
      val w = nonZeroIndices(j)
      val n_hjw = localCounts(w)
      val n_kw = globalCounts(w)
      var s = 1
      while (s <= n_hjw) {
        logNumerator += math.log(gamma + n_kw + s - 1)
        s += 1
      }
      j += 1
    }

    var logDenominator = 0.0
    var s = 1
    while (s <= totalLocalWords) {
      logDenominator += math.log(V * gamma + totalGlobalWords + s - 1)
      s += 1
    }

    logNumerator - logDenominator
  }

  def computeClusterPartitionProbabilities(idx: Int): Array[Double] = {
    val clusterLogProbs = new Array[Double](countGlobalCluster.length)
    var k = 0
    while (k < countGlobalCluster.length) {
      clusterLogProbs(k) = log(countGlobalCluster(k).toDouble) + logPosteriorPredictive(idx, k)
      k += 1
    }
    clusterLogProbs
  }

  def drawMembership(idx: Int): Int = {
    val logProbs = computeClusterPartitionProbabilities(idx)
    val logNewCluster = log(alpha) + priorPredictive(idx)
    val normalized = normalizeLogProbability(logProbs :+ logNewCluster)
    sample(normalized)
  }

  def run(nIter: Int = 1, verbose: Boolean = false):
  Array[Int] = {

    var iter = 0
    while (iter < nIter) {
      var idx = 0
      while (idx < n) {
        removeLocalClusterFromItsGlobalCluster(idx)
        val newCluster = drawMembership(idx)
        partition(idx) = newCluster
        addLocalClusterToGlobalCluster(idx)
        idx += 1
      }
      iter += 1
    }
    partition
  }
}