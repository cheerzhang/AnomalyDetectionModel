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
   
   data_process.calculate_metrics
   :alias: calculate_metrics
   
   data_process.check_valid_tensor_data
   :alias: check_valid_tensor_data

   data_process.graph_scatter
   data_process.graph_multiple_histograms
   data_process.category_hist_graph

   models.AnomalyDetectionModel
   models.AnomalyDetectionNN
   models.TrainAnomalyNN
   models.KMeansModel
   models.ImputerNa
   models.SelectFeatures
   models.LowerStr
   models.LengthStr
   models.LogTransform
   models.NumericDataType


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
