package GS
import org.apache.spark.rdd.RDD
import Common.Tools.convertIteratorToArray

import scala.collection.mutable.ArrayBuffer
import GS.Distributed.{MasterCGS, WorkerCGS}

class DisCGS(dataRDD: RDD[(Int, String)],
             vocabulary: Array[String],
             masterGamma: Double,
             workerGamma: Double,
             nbWorkers: Int,
             n: Int) extends Serializable {

  var partitionEveryIteration: List[Array[Int]] = List(Array.fill(n)(0))

  val workersDPM = dataRDD.mapPartitionsWithIndex { (index, dataIt) =>
    val (indices, docs) = convertIteratorToArray(dataIt)
    Iterator(new WorkerCGS(workerID = index, indices = indices, data = docs, vocabulary = vocabulary, gamma = workerGamma))
  }.persist()

  def run(nIter: Int): Unit = {
    val workerResults = workersDPM.map(w => List(w.run(nIter/2))).persist().reduce(_ ++ _).sortBy(_._1)
    var processedResults = extractSufficientStats(workerResults)
    val masterDPMM = new MasterCGS(processedResults.toArray, vocabulary.length, masterGamma, nbWorkers = nbWorkers)
    val globalPartition = masterDPMM.run(nIter/2)
    val masterToWorkerResults = processedResults.indices.map { i =>
      val (workerID, localCluster, _, _, _) = processedResults(i)
      val globalCluster = globalPartition(i)
      (
        workerID,
        localCluster,
        globalCluster,
      )
    }

    val groupedResults = masterToWorkerResults.groupBy(_._1).view.mapValues(_.toArray).toMap
    val groupedResultsBC = workersDPM.context.broadcast(groupedResults)
    processedResults = extractSufficientStats(workerResults)

    val globalPartitionState = workersDPM
      .map({ worker =>
        val workerSlice = groupedResultsBC.value.getOrElse(worker.workerID, Array.empty)
        worker.updateDPMWithMasterResults(workerSlice)
        worker.getGlobalResults
      })
      .reduce(_ ++ _)
      .sortBy(_._1)
      .map(_._2)
    partitionEveryIteration :+= globalPartitionState.toArray

  }
  private def extractSufficientStats(workerResults: List[(Int, ArrayBuffer[Array[Int]], ArrayBuffer[Int], ArrayBuffer[Int])]): List[(Int, Int, Array[Int], Int, Int)] = {
    workerResults.flatMap { case (workerID, wordStats, totalWords, clusterCounts) =>
      clusterCounts.indices.map { k =>
        (workerID, k, wordStats(k), totalWords(k), clusterCounts(k))
      }
    }
   }
  }