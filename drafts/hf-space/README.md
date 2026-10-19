---
title: Turkish Token Tax
emoji: 📊
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
license: cc-by-sa-4.0
pinned: false
---

# Turkish Token Tax

How many more tokens does Turkish need than English for the same meaning? Measured on 1012 FLORES-200 devtest pairs.

| Tokenizer | TR / EN | TR tokens / word | TR chars / token |
|---|---:|---:|---:|
| berturk | 0.5809 | 1.5616 | 4.9376 |
| xlm-r | 1.0392 | 1.8094 | 4.2614 |
| o200k_base | 1.4311 | 2.1840 | 3.5306 |
| qwen2.5 | 1.6242 | 2.5477 | 3.0265 |
| cl100k_base | 1.9212 | 2.9656 | 2.6000 |

Derived from FLORES-200 devtest (NLLB Team, 2022, https://github.com/facebookresearch/flores), licensed CC-BY-SA-4.0. Only aggregate token counts are included; no corpus text.

Source code: https://github.com/eyupcodes/turkish-token-tax
