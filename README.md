# tfjs-bert-zh
中文bert的nodejs版本

- 支持electron

# model
下载[bert_zh_L-12_H-768_A-12_2](https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2)

[国内地址](https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/2)


<code>
    const { Bert } = require('./main');

    let bert = new Bert({

        modelLocalPath:"./model/bert_zh_L-12_H-768_A-12_2"

    });

</code>
