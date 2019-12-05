# README

Original repository can be found in: \
https://bitbucket.org/luisespinosa/neural_de/src/master/

## Requirements Installation

**Important note: Use python3.6**

Run the command below to install requirements:

```bash
sudo python3.6 -m pip install -r requirements.txt
```

# Train Model

Example command:

```bash
python3.6 src/_main.py -d data/datasets/wcl_datasets_v1.2/ -wv data/embeddings/GoogleNews-vectors-negative300.bin -dep ml
```

# Test Model #

**Important note: Sentences must be line by line.**

Example command:

```python3 src/_definition_retrieval.py -d test/deft_corpus/practice_data_sentences.txt -dep ml -m 100 -wv data/embeddings/GoogleNews-vectors-negative300.bin -p data/models/wcl_ml -t 0.8```

---

### References

[1] Espinosa-Anke, L., Schockaert, S. (2018). Syntactically Aware Neural Architectures for Definition Extraction. NAACL 2018 Short Papers.

[2] Bird, S., Dale, R., Dorr, B. J., Gibson, B., Joseph, M. T., Kan, M. Y., ... & Tan, Y. F. (2008). The acl anthology reference corpus: A reference dataset for bibliographic research in computational linguistics.
