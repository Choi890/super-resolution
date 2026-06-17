# Super Resolution

TensorFlow/Keras image super-resolution project for x2, x3, and x4 upscaling.

The original notebook-only ESPCN example has been modernized with:

- Residual ESPCN model with LR-space residual blocks.
- Fast pixel shuffle through `tf.nn.depth_to_space`.
- Bicubic skip connection so the network learns high-frequency residual detail.
- Charbonnier + SSIM training loss instead of plain MSE.
- PSNR and SSIM validation metrics.
- `tf.data` image pipeline with parallel decoding, prefetching, and augmentation.
- CLI scripts for training, evaluation, and inference.

## Install

Use Python 3.10-3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

For CelebA, point `--image-dir` to the directory that contains files such as
`000001.jpg`. The included `list_eval_partition.csv` is used for train/val/test
splits.

```powershell
python train.py `
  --image-dir D:\datasets\img_align_celeba\img_align_celeba `
  --use-celeba-splits `
  --architecture residual-espcn `
  --scale 4 `
  --hr-size 176 `
  --batch-size 16 `
  --epochs 50 `
  --mixed-precision `
  --xla
```

For a quick smoke run:

```powershell
python train.py `
  --image-dir D:\datasets\img_align_celeba\img_align_celeba `
  --use-celeba-splits `
  --limit-train 128 `
  --limit-val 32 `
  --epochs 1
```

Checkpoints are written to `models/`, for example:

- `models/residual-espcn_x4_best.keras`
- `models/residual-espcn_x4_final.keras`

## Inference

```powershell
python infer.py `
  --input image.jpg `
  --output outputs/image_x4.png `
  --weights models/residual-espcn_x4_best.keras `
  --scale 4
```

`example.py` delegates to the same CLI, so this also works:

```powershell
python example.py --input image.jpg --output outputs/image_x4.png --weights models/residual-espcn_x4_best.keras
```

## Evaluate

```powershell
python evaluate.py `
  --image-dir D:\datasets\img_align_celeba\img_align_celeba `
  --use-celeba-splits `
  --split test `
  --weights models/residual-espcn_x4_best.keras `
  --include-bicubic-baseline
```

## Legacy Notes

`Subpixel.py` and `DataGenerator.py` remain as compatibility wrappers for the
old notebooks. New work should use the CLI scripts and the `super_resolution`
package.
