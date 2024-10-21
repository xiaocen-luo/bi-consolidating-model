## A Bi-consolidating Model for Joint Relational Triple Extraction

This repository contians the source code and datasets for the paper: **A Bi-consolidating Model for Joint Relational Triple Extraction**, Xiaocheng Luo, Yanping Chen, Ruixue Tang, Caiwei Yang, Ruizhang Huang and Yongbin Qin, Neurocomputing-2024.

## Bi-consolidating Model

Based on a two-dimensional sentence representation, a bi-consolidating model is proposed to address this problem by simultaneously reinforcing the local and global semantic features relevant to a relation triple. This model consists of a local consolidation component and a global consolidation component. The first component uses a pixel difference convolution to enhance semantic information of a possible triple representation from adjacent regions and mitigate noise in neighbouring neighbours. The second component strengthens the triple representation based a channel attention and a spatial attention, which has the advantage to learn remote semantic dependencies in a sentence. They are helpful to improve the performance of both entity identification and relation type classification in relation triple extraction.

![tagging](/img/our-model.png)

## Usage

1. **Environment**
   ```shell
   conda create -n your_env_name python=3.8
   conda activate your_env_name
   cd bi-consolidating-model
   pip install -r requirements.txt
   ```

2. **The pre-trained BERT**

    The pre-trained BERT (bert-base-cased) will be downloaded automatically after running the code. Also, you can manually download the pre-trained BERT model and decompress it under `./pre_trained_bert`.


3. **Train the model (take NYT as an example)**

    Modify the second dim of `batch_triple_matrix` in `data_loader.py` to the number of relations, and run

    ```shell
    python train.py --dataset=NYT --batch_size=6 --rel_num=24 
    ```
    The model weights with best performance on dev dataset will be stored in `checkpoint/NYT/`

4. **Evaluate on the test set (take NYT as an example)**

    Modify the `model_name` (line 48) to the name of the saved model, and run 
    ```shell
    python test.py --dataset=NYT --rel_num=24
    ```

    The extracted results will be save in `result/NYT`.


## **Acknowledgment**

We would like to express our gratitude to the [OneRel](https://github.com/China-ChallengeHub/OneRel) and [ODRTE](https://github.com/NingJinzhong/ODRTE). project for providing some code snippets that were used in this project. Their contributions have greatly helped in the development and implementation of our code. We appreciate their efforts and the open-source community for fostering collaboration and knowledge sharing.
