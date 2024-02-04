Quick Start
===========

Installation
-----------------

.. _installation:

You can install quick-anomaly-detector from PyPI using the following command:

.. code-block:: python

   pip install quick-anomaly-detector

PyPI link: https://pypi.org/project/quick-anomaly-detector/


Quick Examples
-----------------

Guissian Based Anomaly Model:

.. code-block:: python

   from quick_anomaly_detector.models import AnomalyDetectionModel

   # Guissian Based Anomaly Detection
   # Load your datasets (X_train, X_val, y_val)
   model = AnomalyDetectionModel()

   # Train the model
   model.train(X_train, X_val, y_val)

   # Predict anomalies in the validation dataset
   anomalies = model.predict(X_val)


.. autosummary::
   :toctree: generated
   :nosignatures:

   quick-anomaly-detector.AnomalyDetectionModel
