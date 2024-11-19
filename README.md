# Project

This is the repository for Multi-field Adaptive Retrieval (mFAR).

## Data processing

Use the official library (`stark-qa`) to download the data. To test the download or to create a smaller, truncated version of the dataset, use the optional `--max_docs` argument. `dataset_name` can be `mag`, `prime`, or `amazon`.

```
python stark_to_trec.py --dataset_name prime --out $TMP_DIR [--max_docs 2000]
python download_queries.py --dataset_name prime --out $TMP_DIR [--max_docs 2000]
```

## Contributing

See CONTRIBUTING.md

## Citation

```
@misc{li2024multifieldadaptiveretrieval,
      title={Multi-Field Adaptive Retrieval},
      author={Millicent Li and Tongfei Chen and Benjamin Van Durme and Patrick Xia},
      year={2024},
      eprint={2410.20056},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2410.20056},
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
