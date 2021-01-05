const fs = require("fs"),
    path = require("path"),
    request = require("request"),
    progress = require('request-progress');;

function init() {
    //创建文件夹目录
    let dirPath = path.join(__dirname, "model");
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath);
        console.log("文件夹创建成功");
        let fileName = "bert_zh_L-12_H-768_A-12_2.tar";
        let url = "https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/2";
        progress(request(url), {
                // throttle: 2000,                    // Throttle the progress event to 2000ms, defaults to 1000ms
                // delay: 1000,                       // Only start to emit after 1000ms delay, defaults to 0ms
                // lengthHeader: 'x-transfer-length'  // Length header to use, defaults to content-length
            })
            .on('progress', function(state) {
                console.log('progress', state);
            })
            .on('error', function(err) {
                // Do something with err
            })
            .on('end', function() {
                // Do something after request finishes
                console.log("文件[" + fileName + "]下载完毕");
            })
            .pipe(fs.createWriteStream(path.join(dirPath, fileName)));

    } else {
        console.log("文件夹已存在");
    }
}

module.exports = {
    init: init
}