import { MnistData } from './data.js';
import * as tf from '@tensorflow/tfjs';


export class Brain {

    loaded = false;
    trained = false;
    data: MnistData;
    model;

    constructor() {
        this.data = new MnistData();
        this.setup();
    }

    async setup() {
        await this.data.load();
        this.createModel();
        await this.train();
        this.loaded = true;
    }

    createModel() {
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

    async train() {
        const learningRate = 0.1;
        const optimizier = tf.train.sgd(learningRate);

        this.model.compile({
          optimizer: optimizier,
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy'],
        });

        // How many examples the model should "see" before making a parameter update.
        const BATCH_SIZE = 64;
        // How many batches to train the model for.
        const TRAIN_BATCHES = 100;

        // Every TEST_ITERATION_FREQUENCY batches, test accuracy over TEST_BATCH_SIZE examples.
        // Ideally, we'd compute accuracy over the whole test set, but for performance
        // reasons we'll use a subset.
        const TEST_BATCH_SIZE = 100;
        const TEST_ITERATION_FREQUENCY = 5;

        for (let i = 0; i < TRAIN_BATCHES; i++) {
          const batch = this.data.nextTrainBatch(BATCH_SIZE);

          let testBatch;
          let validationData;

          if (i % TEST_ITERATION_FREQUENCY === 0) {
            testBatch = this.data.nextTestBatch(TEST_BATCH_SIZE);
            validationData = [
              testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]),
              testBatch.labels
            ];
          }

          const history = await this.model.fit(
            batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),  // xs is ein tensor mit BatchSize x 784
            batch.labels,
            {
              batchSize: BATCH_SIZE,
              validationData,
              epochs: 1
            }
          );

          const loss = history.history.loss[0];
          const accuracy = history.history.acc[0];

          console.log(loss);
        }

        console.log('trained');
        this.trained = true;
    }

    getRandomMnistImage() {
        return this.data.nextTrainBatch(1);
    }

    predict(predictData) {

        predictData = predictData.reshape([-1, 28, 28, 1]);

        const result = this.model.predict(predictData);

        result.print();
        const resultValueArr = Array.from(result.argMax(1).dataSync());
        const resultValue = resultValueArr[0] as number;
        const probability = (result.dataSync())[resultValue];

        return { result: resultValue, probablilty: probability };
      }
}
