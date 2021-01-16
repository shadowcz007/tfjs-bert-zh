const fs = require("fs");
const path = require("path");
const tf = require('@tensorflow/tfjs-node')
const { BertWordPieceTokenizer } = require('@nlpjs/bert-tokenizer')
const KMeans = require("tf-kmeans");

const md5 = require('md5');

// const downloadModel = require("./downloadModel");
// downloadModel.init();

class Bert {

    constructor(opts = {}) {
        this.modelLocalPath = opts.modelLocalPath || path.join(__dirname, 'model/bert_zh_L-12_H-768_A-12_2');
        this.vocabFile = opts.vocabFile || path.join(__dirname, 'assets/vocab.txt');
        this.store = {};
    }

    async init() {
            let vocabContent = fs.readFileSync(this.vocabFile, "utf-8");
            this.model = await tf.node.loadSavedModel(this.modelLocalPath);
            this.tokenizer = new BertWordPieceTokenizer({ vocabContent: vocabContent })
        }
        //去除标点等处理
    tokenizerInit(text) {
        return text.trim().replace(/\s/ig, "").slice(0, 500);
    }
    predict(text) {
            text = this.tokenizerInit(text);
            const wpEncoded = this.tokenizer.encodeQuestion(text);
            // console.log(wpEncoded.length);
            // console.log(wpEncoded.ids);
            // console.log(wpEncoded.attentionMask);
            // console.log(wpEncoded.offsets);
            // console.log(wpEncoded.overflowing);
            // console.log(wpEncoded.specialTokensMask);
            // console.log(wpEncoded.typeIds);
            // console.log(wpEncoded.wordIndexes);
            var xs = this.model.predict({
                input_mask: tf.tensor1d(wpEncoded.attentionMask, "int32").expandDims(0),
                input_type_ids: tf.tensor1d(wpEncoded.typeIds, "int32").expandDims(0),
                input_word_ids: tf.tensor1d(wpEncoded.ids, "int32").expandDims(0)
            });
            //console.log(xs.transformer_encoder)
            return xs.transformer_encoder;
        }
        //清理缓存
    autoClearStore() {
            if ((Object.keys(this.store)).length > 1500) this.store = {};
        }
        //预测并缓存
    predictAndStore(text = null, type = "tensor") {
        let id = md5(`${text}__${type}`);
        if (this.store[id]) return this.store[id];
        this.autoClearStore();
        if (type == "tensor") {
            let v = this.predict(text);
            this.store[id] = v;
            return
        } else {
            let v = (this.predict(text)).dataSync();
            this.store[id] = v;
        }

        return this.store[id]
    }

    cosineSimilarity(vector1XY, vector2XY) {
        let v1DotV2 = 0;
        let absV1 = 0;
        let absV2 = 0;

        vector1XY.forEach((v1, index) => {
            const v2 = vector2XY[index];
            v1DotV2 += v1 * v2;
            absV1 += v1 * v1;
            absV2 += v2 * v2;
        })
        absV1 = Math.sqrt(absV1);
        absV2 = Math.sqrt(absV2);

        return v1DotV2 / (absV1 * absV2);
    }

    cosineDistanceMatching(vector1XY, vector2XY) {
        const cosSimilarity = this.cosineSimilarity(vector1XY, vector2XY);
        //console.log(cosSimilarity)
        return Math.sqrt(2 * (1 - cosSimilarity)) || 0;
    }
    async textsRank(target = "", texts = []) {
        this.vectors = [];
        let vs = [];
        let targetVector = this.predictAndStore(target);
        for (let index = 0; index < texts.length; index++) {
            const w = texts[index];
            vs.push(this.predictAndStore(w));
        };

        // console.log(vs)
        let scores = Array.from(vs, (v, i) => {
            return {
                score: this.cosineSimilarity(targetVector, v),
                word: texts[i],
                index: i
            }
        });
        scores = scores.sort((a, b) => b.score - a.score);
        // console.log(scores);
        this.vectors = vs;
        return scores
    }
}


class Clusters {
    constructor() {}
    kmeans(n = 2, dataset = [
        [2, 2, 2],
        [5, 5, 5],
        [3, 3, 3],
        [4, 4, 4],
        [7, 8, 7]
    ]) {
        const kmeans = new KMeans.default({
            k: n,
            maxIter: 30,
            distanceFunction: KMeans.default.EuclideanDistance
        });
        const ds = tf.tensor(dataset);
        const predictions = kmeans.Train(
            ds
        );

        // console.log("Assigned To ", predictions.arraySync());
        // console.log("Centroids Used are ", kmeans.Centroids().arraySync());
        // console.log("Prediction for Given Value is");
        // kmeans.Predict(tf.tensor([2, 3, 2])).print();
        return predictions.arraySync();
    }
}

module.exports.Bert = Bert;
module.exports.Clusters = Clusters;