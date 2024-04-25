---
license: apache-2.0
---

MambaBit. Bit-level cursed model with vocab size=2

* 4 layers, vocab size=2, embedded size = 4096 float32 parm per bit.

* Training was done on first 8030848 bits of tiny Shakespeare in 10 hours on laptop with 16GB VRAM on 9 batches of 128*8 bit each. Training code included in trainer.ipynb

* To run the model run `python mambabit.py "As sun raised over"`.
Expected output
```
As sun raised over me.

LEONTES:
Now means means me not so much as my father,
In the good many lord, and my father come.

KING RICHARD III:
What is my father come and my father,
In the good lord, and my father come and before his father.

GLOUCESTER:
Now the goes of men, a
```


* Bytes are encoded with most significant bit fed first. Eg '7' = [0, 0, 1, 1, 0, 1, 1, 1], so MSB 0 is being fed first
rather than last as if it was with [1, 1, 1, 0, 1, 1, 0, 0]. Intuition with that is that bits at the beginning change less frequent than in the end, so model will be like "I think I will produce a digit" then "I think I will produce 7" instead of "so I spat something. Should it be a number? a letter? dunno"

* I tried to use BF16 originally, but model went into nan (with default big LR) or gradients were so small weights didn't change(smaller LR). I switched back to F32, however some layers still initialize weight with factor x0.001 as I hoped it
would stop model from going to nan.

