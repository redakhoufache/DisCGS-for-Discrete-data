package Common
import scala.io.Source
import play.api.libs.json.Json
import breeze.linalg.{max, sum}
import breeze.numerics.{exp, log}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.jackson.Serialization.writePretty

import java.io.{File, PrintWriter}
import scala.collection.mutable.ArrayBuffer

object Tools extends java.io.Serializable {

  def saveJson[T <: AnyRef](filename: String, data: T)(implicit formats: org.json4s.Formats): Unit = {
    val jsonString = writePretty(data)
    val pw = new PrintWriter(new File(filename))
    try pw.write(jsonString) finally pw.close()
  }

  def mean(xs: Seq[Double]): Double = if (xs.isEmpty) 0.0 else xs.sum / xs.size
  def stddev(xs: Seq[Double]): Double = {
    if (xs.size <= 1) 0.0
    else {
      val avg = mean(xs)
      math.sqrt(xs.map(x => math.pow(x - avg, 2)).sum / (xs.size - 1))
    }
  }

  def loadDataset(datasetName: String): (Array[String], Array[Int]) = {
    val (corpus, trueLabels) = loadCorpusAndLabels(s"../data/$datasetName/data.txt")
    (corpus, trueLabels)
  }

  def createSparkContext(workers: Int, appName: String): SparkContext = {
   Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
      .setMaster(s"local[$workers]")                    
      .setAppName(s"$appName-${workers}Workers")        
      .set("spark.driver.memory",       "32g")
      .set("spark.driver.maxResultSize","24g")
      .set("spark.executor.memory",     "32g")          
      .set("spark.sql.shuffle.partitions",(workers * 6).toString)
      .set("spark.default.parallelism",  (workers * 6).toString)
      .set("spark.memory.fraction",        "0.8")
      .set("spark.memory.storageFraction", "0.3")
   val sc = new SparkContext(conf)
    sc  
}


  def wordCountPerDocument(corpus: Array[String]): Array[Int] = {
    corpus.map(doc => doc.split("\\s+").count(_.nonEmpty))
  }

  def computeTotalWordsPerCluster(countCluster: ArrayBuffer[Int], partition: Array[Int], docLengths: Array[Int]): Array[Int] = {
    val totals = Array.fill(countCluster.length)(0)
    for (i <- partition.indices) {
      totals(partition(i)) += docLengths(i)
    }
    totals
  }

  def countWordByCluster(corpus: Array[String],
                         vocab: Array[String],
                         partition: Array[Int]): Array[Array[Int]] = {

    val vocabIndex = vocab.zipWithIndex.toMap
    val clusterIds = partition.distinct.sorted
    val clusterToIndex = clusterIds.zipWithIndex.toMap

    val counts = Array.fill(clusterIds.length, vocab.length)(0)

    for (i <- corpus.indices) {
      val clusterIdx = clusterToIndex(partition(i))
      val tokens =corpus(i).split("\\s+")
      tokens.foreach { token =>
        vocabIndex.get(token).foreach { wordIdx =>
          counts(clusterIdx)(wordIdx) += 1
        }
      }
    }
    counts
  }

  def buildVocabulary(corpus: Array[String]): Array[String] = {
    val vocabSet = corpus.flatMap { text =>
      text.split("\\s+")
    }.toSet
    vocabSet.toArray.sorted
  }

  def loadCorpusAndLabels(filepath: String): (Array[String], Array[Int]) = {
    val source = Source.fromFile(filepath, "UTF-8")
    val lines = source.getLines().toArray

    val parsed = lines.flatMap { line =>
      if (line.trim.nonEmpty) {
        try {
          val json = Json.parse(line)
          val maybeText = (json \ "text").asOpt[String]
          val maybeCluster = (json \ "cluster").asOpt[Int]

          (maybeText, maybeCluster) match {
            case (Some(text), Some(cluster)) => Some((text, cluster))
            case _ =>
              println(s"Skipping incomplete line: $line")
              None
          }
        } catch {
          case e: Throwable =>
            println(s"Skipping malformed line: $line")
            println(s"Reason: ${e.getMessage}")
            None
        }
      } else None
    }

    source.close()
    val (texts, clusters) = parsed.unzip
    (texts, clusters)
  }

  def logSumExp(X: Array[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def normalizeLogProbability(probs: Array[Double]): Array[Double] = {
    val LSE = logSumExp(probs)
    probs.map(e => exp(e - LSE))
  }

  def sample(probabilities: Array[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error in sampling")
  }

  def partitionToOrderedCount(membership: Array[Int]): Array[Int] = {
    membership.groupBy(identity).view.mapValues(_.length).toArray.sortBy(_._1).map(_._2)
  }

  def convertIteratorToArray(data: Iterator[(Int, String)]): (Array[Int], Array[String]) = {
    val indexBuffer = scala.collection.mutable.ArrayBuffer[Int]()
    val docBuffer = scala.collection.mutable.ArrayBuffer[String]()

    while (data.hasNext) {
      val (idx, doc) = data.next()
      indexBuffer += idx
      docBuffer += doc
    }

    (indexBuffer.toArray, docBuffer.toArray)
  }

}
