import { Component, ViewChild, ElementRef, OnInit } from '@angular/core';
import { MnistData } from './data.js';
import * as tf from '@tensorflow/tfjs';
import { element } from 'protractor';
import { Brain } from './brain.js';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  isDrawing = false;

  predicted: number;
  label: number;
  loaded = false;
  instandPredict = true;

  predictions: { prediction: number, probability: number };

  @ViewChild('drawingCanvas') drawingCanvas: ElementRef;
  ctx;

  @ViewChild('canvas2') c2: ElementRef;

  brain;

  constructor() {
    this.brain = new Brain();
  }

  ngOnInit() {
    this.setupDrawingCanvas();
  }

  setupDrawingCanvas() {

    // if (window.matchMedia('(max-width: 600px)')) {
    // } else {
    //   this.drawingCanvas.nativeElement.width = 200;
    //   this.drawingCanvas.nativeElement.height = 200;

    //   document.getElementById('drawDiv').style.height = '200px';

    // }

    this.clearDrawingCanvas();


    this.drawingCanvas.nativeElement.onmousedown = () => {
      this.isDrawing = true;
    };
    this.drawingCanvas.nativeElement.onmouseup = () => {
      this.isDrawing = false;
      if (this.instandPredict) {
        this.onPredict();
      }
    };

    this.drawingCanvas.nativeElement.addEventListener('touchmove', (e) => {
      this.isDrawing = true;
      drawing(e);
    });
    this.drawingCanvas.nativeElement.addEventListener('touchend', (e) => {
      if (this.instandPredict) {
        this.onPredict();
      }
    });

    const drawing = (e) => {
      if (this.isDrawing) {
        const x = e.layerX || (e.changedTouches[0].clientX - this.drawingCanvas.nativeElement.getBoundingClientRect().left);
        const y = e.layerY || (e.changedTouches[0].clientY - this.drawingCanvas.nativeElement.getBoundingClientRect().top);

        const radius = 10;
        const color = '#fff';

        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.arc(x, y, radius, 0, Math.PI * 2, false);
        this.ctx.fill();
      }
    };

    this.drawingCanvas.nativeElement.onmousemove = drawing;
  }

  clearDrawingCanvas() {
    this.ctx = this.drawingCanvas.nativeElement.getContext('2d');
    this.ctx.beginPath();
    this.ctx.rect(0, 0, 500, 500);
    this.ctx.fillStyle = 'black';
    this.ctx.fill();
  }

  getDrawingData() {
    const imgData = this.ctx.getImageData(0, 0, 280, 280);

    const ctx2 = this.c2.nativeElement.getContext('2d');
    ctx2.drawImage(this.drawingCanvas.nativeElement, 0, 0, 28, 28);

    return ctx2.getImageData(0, 0, 28, 28);
  }

  draw(data, predicted) {
    const div = document.createElement('div');
    div.className = 'prediction-div card';
    div.style.width = '170px';

    const body = document.createElement('div');
    body.className = 'card-img-top';
    body.style.height = '168px';
    div.appendChild(body);

    const footer = document.createElement('div');
    footer.className = 'card-footer';
    div.appendChild(footer);

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.zoom = '6';

    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);

    const label = document.createElement('div');
    label.innerHTML = 'Predicted: ' + predicted.result + '<br>' + 'Probanility: ' + predicted.probablilty;

    ctx.putImageData(data, 0, 0);

    body.appendChild(canvas);
    footer.appendChild(label);
    document.getElementById('predictionResult').appendChild(div);
  }

  private imageDataToTesor(data) {
    const imgDataArray = [];
    for (let i = 0; i < data.data.length; i += 4) {
      imgDataArray.push((data.data[i] / 255));
    }

    const a = tf.tensor(imgDataArray);
    const b = a.reshape([-1, 28, 28, 1]);

    return b;
  }

  private tensorToImageData(tensor) {
    const tensorData = tensor.dataSync();
    const imgData = this.ctx.createImageData(28, 28);

    for (let i = 0; i < tensorData.length; i++) {
      const j = i * 4;
      imgData.data[j + 0] = tensorData[i] * 255;
      imgData.data[j + 1] = tensorData[i] * 255;
      imgData.data[j + 2] = tensorData[i] * 255;
      imgData.data[j + 3] = 255; // tranparancy
    }

    return imgData;
  }

  onPredict() {
    const data = this.getDrawingData();
    this.imageDataToTesor(data);

    const predicted = this.brain.predict(this.imageDataToTesor(data));
    this.draw(data, predicted);
    this.clearDrawingCanvas();
  }

  onRandom() {
    const data = this.brain.getRandomMnistImage();

    const predicted = this.brain.predict(data.xs);
    this.draw(this.tensorToImageData(data.xs), predicted);
  }
}
