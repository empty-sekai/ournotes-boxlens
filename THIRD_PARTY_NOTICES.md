# Third-party notices

Our Notes BoxLens's own code, documentation and released trained weights use
the MIT license in LICENSE. This does not replace third-party terms or grant
rights to game assets, data or trademarks. No game assets are distributed.

The card encoder starts from torchvision's MobileNetV3-Small ImageNet weights
and is fine-tuned as documented in MODEL_CARD.md. torchvision uses BSD-3-Clause;
its notice is retained in licenses/torchvision-BSD-3-Clause.txt. ImageNet is the
pretraining dataset, not a dataset distributed by this project.

The optional resource preparation process uses the separately installed nnnotes
package at commit bb542af0d7e22a0255a7ee45099a51eb7fbadd12 (MIT), and UnityPy.
Their source, dependencies, credentials and extracted resources are not vendored.
Other dependencies are installed separately and retain their own licenses.

Primary sources:

- https://github.com/pytorch/vision/blob/main/LICENSE
- https://github.com/MetaSekaiLab/nnnotes/blob/bb542af0d7e22a0255a7ee45099a51eb7fbadd12/LICENSE

BanG Dream! and the related game names are used only to identify supported
inputs. This is an independent project; no endorsement is implied.
