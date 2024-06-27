# Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item

This is the implementation of the paper "Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Tesla V100 32GB
  - Cuda: 10.2
  - Driver version: 440.95.01
  - CPU: Intel Xeon Gold 6133
- Software:
  - Python: 3.9.5
  - Pytorch: 1.12.0+cu102

You can conda install the `environment.yml` or pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset and get the LLMs embedding according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<yelp/fashion/beauty>/raw/`. The Yelp dataset can be obtained from [https://www.yelp.com/dataset](https://www.yelp.com/dataset). The fashion and beauty datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews).
2. Conduct the preprocessing code `data/data_process.py` to filter cold-start users and items. After the procedure, you will get the id file  `/data/<yelp/fashion/beauty>/hdanled/id_map.json` and the interaction file  `/data/<yelp/fashion/beauty>/handled/inter_seq.txt`.
3. Convert the interaction file to the format used in this repo by running `data/convert_inter.ipynb`.
4. To get the LLMs embedding for each dataset, please run the jupyter notebooks  `/data/<yelp/fashion/beauty>/get_item_embedding.ipynb` and  `/data/<yelp/fashion/beauty>/get_user_embedding.ipynb`. After the running, you will get the LLMs item embedding file `/data/<yelp/fashion/beauty>/handled/itm_emb_np.pkl` and LLMs user embedding file `/data/<yelp/fashion/beauty>/handled/usr_emb_np.pkl`.
5. For dual-view modeling module, we need to run the jupyter notebook `data/pca.ipynb` to get the dimension-reduced LLMs item embedding for initialization, i.e., `/data/<yelp/fashion/beauty>/handled/pca64_itm_emb_np.pkl`.
6. For retrieval augmented self-distillation, we need to run the jupyter notebook `data/retrieval_users.ipynb` to get the similar user set for each user. The output file in this step is `sim_user_100.pkl`

In conclusion, the prerequisite files to run the code are as follows: `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl` and `sim_user_100.pkl`.

⭐️ To ease the reproducibility of our paper, we also upload all preprocessed files to this [link](https://drive.google.com/file/d/1MpBUjCDLiFIEODTnopSCzDAnS8RzO9aV/view?usp=sharing).

## Run and test

1. You can reproduce all LLM-ESR experiments by running the bash as follows:

```
bash experiments/yelp.bash
bash experiments/fashion.bash
bash experiments/beauty.bash
```

2. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.

## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{liu2024large,
  title={Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item},
  author={Liu, Qidong and Wu, Xian and Zhao, Xiangyu and Wang, Yejing and Zhang, Zijian and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2405.20646},
  year={2024}
}
```

## Thanks

The code refers to the repo [SASRec](https://github.com/kang205/SASRec) and [RLMRec](https://github.com/HKUDS/RLMRec).
