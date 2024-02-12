Welcome to quick-anomaly-detector's documentation!
==================================================

**quick-anomaly-detector** is a Python library that provides a collection of classes to model an anomaly detection task.

Check out the :doc:`quick_start` section for further information.

.. toctree::
   :maxdepth: 10

   quick_start

.. currentmodule:: quick_anomaly_detector

.. autosummary::
   :toctree: my_doc


   data_process.apply_transformations
      :alias: apply_transformations
   quick_anomaly_detector.data_process.calculate_metrics
      :alias: calculate_metrics
   quick_anomaly_detector.data_process.check_valid_tensor_data

   quick_anomaly_detector.data_process.graph_scatter
   quick_anomaly_detector.data_process.graph_multiple_histograms
   quick_anomaly_detector.data_process.category_hist_graph

   quick_anomaly_detector.models.AnomalyDetectionModel
   quick_anomaly_detector.models.AnomalyDetectionNN
   quick_anomaly_detector.models.TrainAnomalyNN
   quick_anomaly_detector.models.KMeansModel
   quick_anomaly_detector.models.ImputerNa
   quick_anomaly_detector.models.SelectFeatures
   quick_anomaly_detector.models.LowerStr
   quick_anomaly_detector.models.LengthStr
   quick_anomaly_detector.models.LogTransform
   quick_anomaly_detector.models.NumericDataType


Additional Information
----------------------

This is an additional section with more information about your project.

Usage Example
~~~~~~~~~~~~~

Here you can provide code examples, explanations, or anything else you find relevant.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
