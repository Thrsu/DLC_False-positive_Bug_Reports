# DLC_False-positive_Bug_Reports

This repository contains curated datasets and experimental code for the study: **"False-Positive Bug Reports in Deep Learning Compilers: Stages, Root Causes, and Mitigation"**. It provides annotated bug reports from [TVM](https://github.com/apache/tvm) and [OpenVINO](https://github.com/openvinotoolkit/openvino), and includes code for LLM-based false-positive bug report classification.

## Dataset

Each dataset `.xlsx` file contains labeled bug reports with the following fields:

- **Title**: The title of the GitHub issue/discussion.
- **Link**: A URL link to the original GitHub entry.
- **Type**: One of `{FalsePositive, Confirmed}` indicating whether the report refers to a genuine bug.

The False-positive bug reports contain the stage and root cause labels:

- **Stage**: The DL compiler stage where the report occurs (e.g., Build and Import, Model Loading, etc.).
- **Root Cause**: High-level cause category (e.g., Incorrect Usage, Incorrect Environment Configuration).
- **Sub Root Cause**: Fine-grained categorization (e.g., API Misuse, DL Software Configuration).

## LLM

The `LLM` folder contains code used to classify bug reports via few-shot prompting using GPT-4o. You can modify the examples or plug in your own data by editing the code.