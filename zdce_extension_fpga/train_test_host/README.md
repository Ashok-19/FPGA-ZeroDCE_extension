

# Prerequisites:

* Refer  [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git)




# Training the model


If you plan on retrain the model , dont train it from scratch. Instead, make the load_pretrain = True in [low_light_train.py]()

                load_pretrain = True

In the model path, provide the pretrained model's path.

This introduces Transfer learning, effectively making the model much more robust.


Refer [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) for training and testing