Configuration Overview

All experiment configuration files follow a shared structure and differ primarily in the backbone and neck components.
All remaining settings (data pipeline, optimization strategy, training schedule, and evaluation protocol) are kept consistent to ensure fair and comparable experiments.

For reference and documentation purposes, the configuration file
configs/faster_rcnn_h0_midogpp_explained.py
is intentionally more extensively commented. It explains key MMDetection concepts and design choices (e.g., feature pyramid construction, anchor configuration, and training parameters).

All other configuration files are kept minimal and concise, focusing only on the settings that differ between experiments, in order to improve readability and maintainability.