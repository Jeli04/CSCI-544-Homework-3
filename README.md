# CSCI 544 HW3 

## Task 1: Simple BiLSTM

```bash
python train_blstm.py --config blstm1.json
```
- Optimizer: SGD, lr=0.1, momentum=0.9
- Scheduler: ReduceLROnPlateau (factor=0.1)
- No pre-trained embeddings

Saves the best checkpoint to `blstm.pt`. Make sure to rename into `blstm1.pt`.

> Note: learning rate needs to be aggressive. ReduceLROnPlateau works best for Task 1.

---

## Task 2: BiLSTM with GloVe embeddings

```bash
python train_blstm.py --config blstm2.json
```
- Optimizer: Adam, lr=0.001, momentum=0.9
- Scheduler: ReduceLROnPlateau (factor=0.3)
- No pre-trained embeddings

Saves the best checkpoint to `blstm.pt`. Make sure to rename into `blstm2.pt`.

---

## Bonus: CNN-BiLSTM-CRF

```bash
python train_cnn_blstm.py --config cnn_blstm.json
```
- Char-level CNN + word-level BiLSTM + CRF decoder
- Optimizer: SGD, lr=0.015, momentum=0.9
- Scheduler: LambdaLR with decay `lr / (1 + rho * epoch)`, rho=0.05
- Pre-trained GloVe embeddings 
- Saves best checkpoint to `blstm_cnn.pt`


---

## Inference

Run inference for Task 1 (blstm1):
```bash
python inference.py --checkpoint blstm1.pt --config blstm1.json
```

Run inference for Task 2 (blstm2):
```bash
python inference.py --checkpoint blstm2.pt --config blstm2.json
```

Run inference for Bonus (CNN-BiLSTM-CRF):
```bash
python inference.py --checkpoint models/blstm_cnn.pt --config configs/cnn_blstm.json --model cnn_blstm
```

Outputs: `dev.out`, `test.out`. Make sure to properly rename.
