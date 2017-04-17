
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("Word Count")
sc = SparkContext(conf = conf)

rdd = sc.textFile("<path_to_books>")
words = rdd.flatMap(lambda x: x.split())
result = words.countByValue()