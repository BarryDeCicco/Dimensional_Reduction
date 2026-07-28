# Dimensional_Reduction
This is a playground for dimensional reduction for eventual anomaly detection.

Isolation Forests have a problem with high cardinality variables, such as diagnostic/procedure codes.

There are a methods to deal with that, such as:

1) Using hierarchies in ICD-10 codes - see Clinical Classifications Software Refined (CCSR). See https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
2) Some sort of vector-based system:
  a) Med2Vec.  See https://medium.com/codex/med2vec-transforming-healthcare-with-data-driven-insights-987120317e80 and https://github.com/mp2893/med2vec
  b) https://github.com/ncbi-nlp/BioSentVec

In this repository, I will try to carry out some of these methods - first, in Google Colab, in Jupyter notebooks; later in AWS.

