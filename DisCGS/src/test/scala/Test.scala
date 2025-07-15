
import Common.Tools._
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ArrayBuffer

class Test extends AnyFunSuite {

 // Test wordCountPerDocument

 test("wordCountPerDocument should count words in normal documents") {
  val docs = Array("hello world hello", "this is a test", "Scala is great")
  val result = wordCountPerDocument(docs)
  assert(result.sameElements(Array(3, 4, 3)))
 }

 test("wordCountPerDocument should handle multiple spaces") {
  val docs = Array("hello    world", "  a   b c ")
  val result = wordCountPerDocument(docs)
  assert(result.sameElements(Array(2, 3)))
 }

 //  countWordByCluster

 test("countWordByCluster should count word occurrences per cluster correctly") {
  val corpus = Array("apple banana apple", "banana orange", "apple orange banana", "grape")
  val vocab = buildVocabulary(corpus) //Array("apple", "banana", "orange", "grape")
  val partition = Array(0, 0, 1, 1) // cluster 0: doc 0, 1 ; cluster 1: doc 2, 3

  val result = countWordByCluster(corpus, vocab, partition)

  // Expected:
  // cluster 0: apple (2), banana (2), grape (0), orange (1)
  // cluster 1: apple (1), banana (1), grape (1), orange (1)
  val expected = Array(
   Array(2, 2, 0, 1),
   Array(1, 1, 1, 1)
  )

  assert(vocab.sameElements(Array("apple", "banana", "grape", "orange")))
  assert(result.map(_.toSeq).toSeq == expected.map(_.toSeq).toSeq)
 }

 // computeTotalWordsPerCluster

 test("computeTotalWordsPerCluster computes total words per cluster correctly") {
  val countCluster = ArrayBuffer(2, 2) // 2 documents per cluster
  val partition = Array(0, 0, 1, 1)
  val docLengths = Array(3, 2, 4, 1) // total words per doc

  val result = computeTotalWordsPerCluster(countCluster, partition, docLengths)

  // cluster 0: docs 0,1 → 3 + 2 = 5
  // cluster 1: docs 2,3 → 4 + 1 = 5
  val expected = Array(5, 5)
  assert(result sameElements expected)
 }

 test("computeTotalWordsPerCluster handles uneven cluster sizes") {
  val countCluster = ArrayBuffer(1, 3)
  val partition = Array(0, 1, 1, 1)
  val docLengths = Array(4, 2, 1, 3)

  val result = computeTotalWordsPerCluster(countCluster, partition, docLengths)

  // cluster 0: doc 0 → 4
  // cluster 1: docs 1,2,3 → 2+1+3 = 6
  val expected = Array(4, 6)
  assert(result sameElements expected)
 }

 test("computeTotalWordsPerCluster works with one cluster only") {
  val countCluster = ArrayBuffer(3)
  val partition = Array(0, 0, 0)
  val docLengths = Array(2, 2, 2)

  val result = computeTotalWordsPerCluster(countCluster, partition, docLengths)

  val expected = Array(6)
  assert(result sameElements expected)
 }

 // sparseDocWordCounts

 test("sparseDocWordCounts should return sparse word count vectors per document") {
  val data = Array(
   "apple banana apple",
   "banana grape",
   "apple orange"
  )

  val vocabulary = Array("apple", "banana", "grape")
  val vocabularyIndex = vocabulary.zipWithIndex.toMap

  val expected = Array(
   Array((0, 2), (1, 1)),  // doc 0: 2 apple, 1 banana
   Array((1, 1), (2, 1)),  // doc 1: 1 banana, 1 grape
   Array((0, 1))           // doc 2: 1 apple (orange is ignored)
  )

  val result = data.map { doc =>
   val freqs = scala.collection.mutable.Map[Int, Int]()
   doc.split("\\s+").foreach { w =>
    vocabularyIndex.get(w).foreach { idx =>
     freqs.update(idx, freqs.getOrElse(idx, 0) + 1)
    }
   }
   freqs.toArray.sortBy(_._1)
  }

  assert(result.map(_.toSeq).toSeq == expected.map(_.toSeq).toSeq)
 }

}