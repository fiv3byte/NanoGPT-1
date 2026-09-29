# NanoGPT - Обучение на меньших устройствах


- **~1.04M параметров** (dim=128, 5 слоёв, 4 головы, 4× MLP + weight tying)
- **Каузальный attention** (модель «смотрит в будущее»)
- Оптимизации под **Quadro P3000 (6 ГБ)** / GTX 1060 / RTX 3050 и т.п.
- Mixed precision (AMP), cosine LR, gradient clipping
- Квантизация int8 после обучения (~3 МБ)
- Обучение на **байтах**
## Быстрый старт

```bash
pip install -r requirements.txt

# обучение (по умолчанию ~1.04M, block=128, batch=32)
python train.py

# генерация
python generate.py --prompt "Жил-был" --max_tokens 250
```

### Подстройка под вашу карту

```bash
# больше контекста / батч (P3000 спокойно тянет)
python train.py --block_size 192 --batch_size 48 --epochs 10

# ещё меньше модель (~0.6M) — если совсем туго
python train.py --dim 96 --n_layers 6 --batch_size 64

# чуть больше (~1.3M)
python train.py --dim 160 --n_layers 4 --batch_size 24
```

### Свои данные

Положите любой `.txt` (UTF-8) в `data/input.txt`. Если файла нет — используется встроенная сказка про Петю Зубова.

```bash
# пример: свой текст
cp my_book.txt data/input.txt
python train.py --epochs 12 --batch_size 40
```

## Архитектура (по умолчанию)

```
vocab_size   = 256          # байты
block_size   = 128          # контекст
dim          = 128
n_layers     = 5
n_heads      = 4
MLP          = 4×           # как в GPT-2
weight tying = да           # head.weight = tok_emb.weight
≈ 1.04M параметров
```

Размер в fp32 ≈ 4 МБ, в int8 ≈ 1.2–1.5 МБ.

## Что исправлено

1. **Causal mask** — без неё модель не является авторегрессивной.
2. **Стандартный 4× MLP** вместо 2×.
3. **Weight tying** — меньше параметров, лучше качество.
4. **Правильная инициализация** + cosine schedule.
5. **AMP** + pin_memory — ускорение на CUDA.
6. Чекпоинты сохраняют конфиг → `generate.py` поднимает модель автоматически.

## Лицензия

GPL-3.0 (Только для образовательных целей).
