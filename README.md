# inf1test
# 参考
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/pytorch-setup/pytorch-install.html#install-neuron-pytorch
# 感想
inf1 instanceを使用してチュートリアルなどはうまくいったが、肝心のmodelをneuronにコンパイルすると帰ってくる値がnanのテンソルになってしまう現象に見舞われた
tokenizerにpaddingを施すと結果がnanになってしまうことが分かっている。
チュートリアルのモデルでは、paddingできていたのでなにかカラクリがありそうだけどあきらめた
# 使ったものに関して
AMI: Deep Learning AMI GPU PyTorch 1.11.0 (Ubuntu 20.04) 20220927
Instance Type: inf1.2xlarge
