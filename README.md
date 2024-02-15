# LABEL-GUIDED CORESET GENERATION FOR COMPUTATIONALLY EFFICIENT CHEST X-RAY DIAGNOSIS
This is the official PyTorch implementation for the LCG method.

[Paper](#) | [Model](#model) | [Data](#data)<br>
**Label-Guided Coreset Generation for Computationally Efficient Chest X-ray Diagnosis**<br>
by **Jayant Mahawar**, Bhargab Chattopadhyay, Angshuman Paul<br>

<object data="./figures/WorkFlow.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="figures/WorkFlow.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="figures/WorkFlow.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Abstract
Training a deep learning model with a large dataset requires a large number of computations. While deploying such a system in a real-time scenario that demands rapid training, substantial computational resources may be required. Such resources might be elusive on many occasions. To address this challenge, we propose a strategy to reduce the size of the training dataset utilizing the class labels of the training data without compromising the quality of the training for downstream tasks. Our approach may reduce the demand for computational resources. The smaller version of the training dataset, known as a coreset, ensures a comprehensive representation of all the classes in the dataset. Our algorithm also handles class imbalance utilizing the class labels of the data. We evaluate the algorithm's performance for the downstream task of multi-label classification of chest x-rays. Rigorous experiments on various publicly available datasets demonstrate our ability to reduce dataset size by almost 1/3rd and address class imbalance while maintaining the classification model's performance.

## Release Notes
This repository is a faithful reimplementation of HAM in PyTorch, including all the training and evaluation codes on NIH Chest X-ray14 dataset.
