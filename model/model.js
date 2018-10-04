const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const data = require('./data');

let model;

main();

async function main() {
    await data.loadData();
    console.log('__');
    createModel();
    console.log('***');
    await train();
}

function createModel() {
    this.model = tf.sequential();

    this.model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 8,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    this.model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    this.model.add(tf.layers.flatten());

    this.model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }));
}

async function train() {
    const learningRate = 0.01;
    const optimizier = tf.train.sgd(learningRate);

    this.model.compile({
      optimizer: optimizier,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    
    const trainingData = data.getTrainData();
    
    await this.model.fit(trainingData.images, trainingData.labels, {
        epochs: 8,
        batchSize: 100,
        validationSplit: 0.15,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                console.log(logs.loss);
                await tf.nextFrame();
            }
        }
    });

    const validationData = data.getTestData();
    const evalResult = this.model.evaluate(validationData.images, validationData.labels);

    console.log('EvaluationResult: \n' 
    + `  Loss = ${evalResult[0].dataSync()[0].toFixed(3)}; ` 
    + `  Accuracy = ${evalResult[1].dataSync()[0].toFixed(3)}`);

    console.log('trained');

    const saveResult = await this.model.save('file://./src/assets/model/model-1a');
    console.log(saveResult);
}