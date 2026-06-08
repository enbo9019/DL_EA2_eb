document.addEventListener("DOMContentLoaded", async () => {
  const statusElement = document.getElementById("status");

  if (typeof tf === "undefined") {
    statusElement.textContent = "Fehler: TensorFlow.js wurde nicht geladen.";
    return;
  }

  if (typeof Plotly === "undefined") {
    statusElement.textContent = "Fehler: Plotly wurde nicht geladen.";
    return;
  }

  await tf.ready();

  /*
    Zentrale Einstellungen der Aufgabe.
    cleanModelEpochs: Modell ohne Rauschen.
    bestFitEpochs: nicht zu langes Training auf verrauschten Daten.
    overfitEpochs: langes Training, um Overfitting sichtbar zu machen.
  */
  const config = {
    numberOfSamples: 100,
    xMin: -2,
    xMax: 2,
    trainRatio: 0.5,
    noiseVariance: 0.05,
    seed: 42,

    cleanModelEpochs: 300,
    bestFitEpochs: 250,
    overfitEpochs: 2500,

    batchSize: 32,
    learningRate: 0.01
  };

  // Datensätze erzeugen: clean/noisy und jeweils train/test.
  const data = createDataSets(config);

  // R1: Datensätze darstellen.
  plotDataSet(
    "plot-clean",
    data.cleanTrain,
    data.cleanTest,
    "Unverrauschte Daten"
  );

  plotDataSet(
    "plot-noisy",
    data.noisyTrain,
    data.noisyTest,
    "Verrauschte Daten"
  );

  /*
    Wichtig:
    Die Testdaten werden hier nicht zum Training benutzt.
    Sie werden nur nach dem Training für den MSE berechnet und visualisiert.
  */

  // ------------------------------------------------------------
  // Modell 1: Training ohne Rauschen
  // ------------------------------------------------------------
  statusElement.textContent = "Modell ohne Rauschen wird trainiert ...";

  const cleanModel = createRegressionModel(config.learningRate, 1001);

  const cleanTrainingResult = await trainModel(
  cleanModel,
  data.cleanTrain,
  config.cleanModelEpochs,
  config.batchSize,
  statusElement,
  "Clean-Modell",
  50
);

  const cleanTrainLoss = calculateMSE(cleanModel, data.cleanTrain);
  const cleanTestLoss = calculateMSE(cleanModel, data.cleanTest);

  plotPrediction(
    "plot-clean-train-prediction",
    cleanModel,
    data.cleanTrain,
    "Vorhersage auf Trainingsdaten ohne Rauschen",
    "Trainingsdaten",
    "#2563eb"
  );

  plotPrediction(
    "plot-clean-test-prediction",
    cleanModel,
    data.cleanTest,
    "Vorhersage auf Testdaten ohne Rauschen",
    "Testdaten",
    "#dc2626"
  );

  document.getElementById("loss-clean-train").textContent =
    `Loss Training ohne Rauschen (MSE): ${cleanTrainLoss.toFixed(6)}`;

  document.getElementById("loss-clean-test").textContent =
    `Loss Test ohne Rauschen (MSE): ${cleanTestLoss.toFixed(6)}`;

  // ------------------------------------------------------------
  // Modell 2: Best-Fit auf verrauschten Daten
  // ------------------------------------------------------------
  statusElement.textContent = "Best-Fit-Modell mit Rauschen wird trainiert ...";

  const bestFitModel = createRegressionModel(config.learningRate, 2001);

  const bestFitTrainingResult = await trainModel(
  bestFitModel,
  data.noisyTrain,
  config.bestFitEpochs,
  config.batchSize,
  statusElement,
  "Best-Fit-Modell",
  50
);

  const bestTrainLoss = calculateMSE(bestFitModel, data.noisyTrain);
  const bestTestLoss = calculateMSE(bestFitModel, data.noisyTest);

  plotPrediction(
    "plot-best-train-prediction",
    bestFitModel,
    data.noisyTrain,
    "Best-Fit-Vorhersage auf verrauschten Trainingsdaten",
    "Trainingsdaten",
    "#2563eb"
  );

  plotPrediction(
    "plot-best-test-prediction",
    bestFitModel,
    data.noisyTest,
    "Best-Fit-Vorhersage auf verrauschten Testdaten",
    "Testdaten",
    "#dc2626"
  );

  document.getElementById("loss-best-train").textContent =
    `Loss Training mit Rauschen, Best-Fit (MSE): ${bestTrainLoss.toFixed(6)}`;

  document.getElementById("loss-best-test").textContent =
    `Loss Test mit Rauschen, Best-Fit (MSE): ${bestTestLoss.toFixed(6)}`;

  // ------------------------------------------------------------
  // Modell 3: Overfit auf verrauschten Daten
  // ------------------------------------------------------------
  statusElement.textContent = "Overfit-Modell mit Rauschen wird trainiert ...";

  const overfitModel = createRegressionModel(config.learningRate, 3001);

  const overfitTrainingResult = await trainModel(
  overfitModel,
  data.noisyTrain,
  config.overfitEpochs,
  config.batchSize,
  statusElement,
  "Overfit-Modell",
  250
);

  const overfitTrainLoss = calculateMSE(overfitModel, data.noisyTrain);
  const overfitTestLoss = calculateMSE(overfitModel, data.noisyTest);

  plotPrediction(
    "plot-overfit-train-prediction",
    overfitModel,
    data.noisyTrain,
    "Overfit-Vorhersage auf verrauschten Trainingsdaten",
    "Trainingsdaten",
    "#2563eb"
  );

  plotPrediction(
    "plot-overfit-test-prediction",
    overfitModel,
    data.noisyTest,
    "Overfit-Vorhersage auf verrauschten Testdaten",
    "Testdaten",
    "#dc2626"
  );

  document.getElementById("loss-overfit-train").textContent =
    `Loss Training mit Rauschen, Overfit (MSE): ${overfitTrainLoss.toFixed(6)}`;

  document.getElementById("loss-overfit-test").textContent =
    `Loss Test mit Rauschen, Overfit (MSE): ${overfitTestLoss.toFixed(6)}`;

// R5: Loss-Verläufe und finaler MSE-Vergleich visualisieren.
plotTrainingLoss(
  "plot-training-loss",
  [
    {
      name: "Clean-Modell",
      losses: cleanTrainingResult.history.loss
    },
    {
      name: "Best-Fit-Modell",
      losses: bestFitTrainingResult.history.loss
    },
    {
      name: "Overfit-Modell",
      losses: overfitTrainingResult.history.loss
    }
  ]
);

plotFinalLossComparison(
  "plot-final-loss",
  [
    {
      modelName: "Clean",
      trainLoss: cleanTrainLoss,
      testLoss: cleanTestLoss
    },
    {
      modelName: "Best-Fit",
      trainLoss: bestTrainLoss,
      testLoss: bestTestLoss
    },
    {
      modelName: "Overfit",
      trainLoss: overfitTrainLoss,
      testLoss: overfitTestLoss
    }
  ]
);

  renderResultSummary([
  {
    modelName: "Clean-Modell",
    trainLoss: cleanTrainLoss,
    testLoss: cleanTestLoss,
    interpretation: "Gute Approximation ohne Label-Rauschen"
  },
  {
    modelName: "Best-Fit-Modell",
    trainLoss: bestTrainLoss,
    testLoss: bestTestLoss,
    interpretation: "Gute Generalisierung trotz verrauschter Daten"
  },
  {
    modelName: "Overfit-Modell",
    trainLoss: overfitTrainLoss,
    testLoss: overfitTestLoss,
    interpretation: "Training-Loss deutlich kleiner als Test-Loss"
  }
]);

statusElement.textContent =
  "Training abgeschlossen. Alle Datensätze, Vorhersagen und Loss-Werte wurden berechnet.";

  console.log("TensorFlow.js Version:", tf.version.tfjs);
  console.log("Datensätze:", data);
  console.log("Loss Clean:", {
    train: cleanTrainLoss,
    test: cleanTestLoss
  });
  console.log("Loss Best-Fit:", {
    train: bestTrainLoss,
    test: bestTestLoss
  });
  console.log("Loss Overfit:", {
    train: overfitTrainLoss,
    test: overfitTestLoss
  });
});

function renderResultSummary(results) {
  /*
    Füllt die Ergebnistabelle im Statusbereich.
    Die Werte werden nicht hart codiert, sondern aus den berechneten MSE-Werten erzeugt.
  */
  const summaryContainer = document.getElementById("result-summary");
  const tableBody = document.getElementById("summary-table-body");

  tableBody.innerHTML = "";

  results.forEach(result => {
    const row = document.createElement("tr");

    row.innerHTML = `
      <td>${result.modelName}</td>
      <td>${result.trainLoss.toFixed(6)}</td>
      <td>${result.testLoss.toFixed(6)}</td>
      <td>${result.interpretation}</td>
    `;

    tableBody.appendChild(row);
  });

  summaryContainer.classList.remove("hidden");
}

function targetFunction(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}


function createDataSets(config) {
  const random = mulberry32(config.seed);
  const cleanData = [];

  // 100 x-Werte gleichverteilt aus [-2, +2] erzeugen.
  for (let i = 0; i < config.numberOfSamples; i++) {
    const x = randomUniform(config.xMin, config.xMax, random);
    const y = targetFunction(x);

    cleanData.push({ x, y });
  }

  // Zufälliger Train/Test-Split.
  shuffleArray(cleanData, random);

  const trainSize = Math.floor(config.numberOfSamples * config.trainRatio);

  const cleanTrain = cleanData.slice(0, trainSize);
  const cleanTest = cleanData.slice(trainSize);

  // Varianz V=0.05 bedeutet Standardabweichung sqrt(0.05).
  const noiseStdDev = Math.sqrt(config.noiseVariance);

  // Nur y wird verrauscht. x bleibt unverändert.
  const noisyTrain = cleanTrain.map(point => ({
    x: point.x,
    y: point.y + randomGaussian(0, noiseStdDev, random)
  }));

  const noisyTest = cleanTest.map(point => ({
    x: point.x,
    y: point.y + randomGaussian(0, noiseStdDev, random)
  }));

  return {
    cleanTrain,
    cleanTest,
    noisyTrain,
    noisyTest
  };
}


function createRegressionModel(learningRate, seed) {
  const model = tf.sequential();

  /*
    Netzwerkarchitektur laut Aufgabenstellung:
    - Input: ein x-Wert
    - 2 Hidden Layer mit je 100 Neuronen
    - ReLU in den Hidden Layern
    - linearer Output für Regression
  */
  model.add(tf.layers.dense({
    inputShape: [1],
    units: 100,
    activation: "relu",
    kernelInitializer: tf.initializers.glorotUniform({ seed }),
    biasInitializer: tf.initializers.zeros()
  }));

  model.add(tf.layers.dense({
    units: 100,
    activation: "relu",
    kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
    biasInitializer: tf.initializers.zeros()
  }));

  model.add(tf.layers.dense({
    units: 1,
    activation: "linear",
    kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 2 }),
    biasInitializer: tf.initializers.zeros()
  }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "meanSquaredError"
  });

  return model;
}


async function trainModel(
  model,
  trainData,
  epochs,
  batchSize,
  statusElement,
  modelName,
  logEvery
) {
  const trainXs = tf.tensor2d(
    trainData.map(point => point.x),
    [trainData.length, 1]
  );

  const trainYs = tf.tensor2d(
    trainData.map(point => point.y),
    [trainData.length, 1]
  );

  const result = await model.fit(trainXs, trainYs, {
    epochs,
    batchSize,
    shuffle: true,

    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const currentEpoch = epoch + 1;

        if (
          currentEpoch === 1 ||
          currentEpoch % logEvery === 0 ||
          currentEpoch === epochs
        ) {
          const message =
            `${modelName}: Epoche ${currentEpoch}/${epochs}, ` +
            `Training Loss = ${logs.loss.toFixed(6)}`;

          statusElement.textContent = message;
          console.log(message);
        }

        // Gibt dem Browser Zeit, die Oberfläche während des Trainings zu aktualisieren.
        await tf.nextFrame();
      }
    }
  });

  trainXs.dispose();
  trainYs.dispose();

  return result;
}


function calculateMSE(model, dataSet) {
  return tf.tidy(() => {
    const xs = tf.tensor2d(
      dataSet.map(point => point.x),
      [dataSet.length, 1]
    );

    const ys = tf.tensor2d(
      dataSet.map(point => point.y),
      [dataSet.length, 1]
    );

    const predictions = model.predict(xs);
    const mseTensor = predictions.sub(ys).square().mean();

    return mseTensor.dataSync()[0];
  });
}


function predictCurve(model, xMin = -2, xMax = 2, numberOfPoints = 200) {
  const xs = [];

  for (let i = 0; i < numberOfPoints; i++) {
    const x = xMin + (i / (numberOfPoints - 1)) * (xMax - xMin);
    xs.push(x);
  }

  const ys = tf.tidy(() => {
    const inputTensor = tf.tensor2d(xs, [xs.length, 1]);
    const predictionTensor = model.predict(inputTensor);
    return Array.from(predictionTensor.dataSync());
  });

  return xs.map((x, index) => ({
    x,
    y: ys[index]
  }));
}


function plotDataSet(plotId, trainData, testData, title) {
  const sortedTrain = [...trainData].sort((a, b) => a.x - b.x);
  const sortedTest = [...testData].sort((a, b) => a.x - b.x);

  const trainTrace = {
    x: sortedTrain.map(point => point.x),
    y: sortedTrain.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Training",
    marker: {
      color: "#2563eb",
      size: 8
    }
  };

  const testTrace = {
    x: sortedTest.map(point => point.x),
    y: sortedTest.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Test",
    marker: {
      color: "#dc2626",
      size: 8
    }
  };

  Plotly.newPlot(
    plotId,
    [trainTrace, testTrace],
    createPlotLayout(title),
    createPlotOptions()
  );
}


function plotPrediction(plotId, model, dataSet, title, dataName, dataColor) {
  const sortedData = [...dataSet].sort((a, b) => a.x - b.x);
  const predictionCurve = predictCurve(model);

  const dataTrace = {
    x: sortedData.map(point => point.x),
    y: sortedData.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: dataName,
    marker: {
      color: dataColor,
      size: 8
    }
  };

  const predictionTrace = {
    x: predictionCurve.map(point => point.x),
    y: predictionCurve.map(point => point.y),
    mode: "lines",
    type: "scatter",
    name: "Modellvorhersage",
    line: {
      color: "#16a34a",
      width: 3
    }
  };

  /*
    Die Ground Truth wird zusätzlich als Orientierung angezeigt.
    Trainiert wird aber nur mit den erzeugten Datenpunkten.
  */
  const groundTruthTrace = {
    x: predictionCurve.map(point => point.x),
    y: predictionCurve.map(point => targetFunction(point.x)),
    mode: "lines",
    type: "scatter",
    name: "Ground Truth",
    line: {
      color: "#111827",
      width: 2,
      dash: "dot"
    }
  };

  Plotly.newPlot(
    plotId,
    [dataTrace, predictionTrace, groundTruthTrace],
    createPlotLayout(title),
    createPlotOptions()
  );
}

function plotTrainingLoss(plotId, modelHistories) {
  /*
    Der Overfit-Verlauf hat sehr viele Epochen.
    Deshalb werden bei langen Verläufen nicht alle Punkte geplottet,
    sondern sinnvoll ausgedünnt. Das hält das Diagramm übersichtlich.
  */
  const traces = modelHistories.map(history => {
    const losses = history.losses;
    const stepSize = Math.max(1, Math.floor(losses.length / 300));

    const epochs = [];
    const plottedLosses = [];

    for (let i = 0; i < losses.length; i += stepSize) {
      epochs.push(i + 1);
      plottedLosses.push(losses[i]);
    }

    return {
      x: epochs,
      y: plottedLosses,
      mode: "lines",
      type: "scatter",
      name: history.name
    };
  });

  const layout = {
  title: "Trainings-Loss über die Epochen",
  xaxis: {
    title: {
      text: "Epoche",
      standoff: 12
    }
  },
  yaxis: {
    title: "MSE / Loss",
    type: "log"
  },
  margin: {
    l: 65,
    r: 25,
    t: 50,
    b: 105
  },
  legend: {
    orientation: "h",
    x: 0.5,
    xanchor: "center",
    y: -0.32
  }
};

  Plotly.newPlot(plotId, traces, layout, createPlotOptions());
}


function plotFinalLossComparison(plotId, lossResults) {
  /*
    Dieses Balkendiagramm zeigt direkt den Unterschied zwischen
    Trainings-Loss und Test-Loss. Beim Overfit-Modell sollte der
    Trainings-Loss kleiner als der Test-Loss sein.
  */
  const modelNames = lossResults.map(result => result.modelName);

  const trainTrace = {
    x: modelNames,
    y: lossResults.map(result => result.trainLoss),
    type: "bar",
    name: "Training MSE"
  };

  const testTrace = {
    x: modelNames,
    y: lossResults.map(result => result.testLoss),
    type: "bar",
    name: "Test MSE"
  };

  const layout = {
  title: "Finaler Vergleich von Training- und Test-MSE",
  xaxis: {
    title: {
      text: "Modell",
      standoff: 12
    }
  },
  yaxis: {
    title: "MSE"
  },
  barmode: "group",
  margin: {
    l: 65,
    r: 25,
    t: 50,
    b: 105
  },
  legend: {
    orientation: "h",
    x: 0.5,
    xanchor: "center",
    y: -0.32
  }
};

  Plotly.newPlot(plotId, [trainTrace, testTrace], layout, createPlotOptions());
}

function createPlotLayout(title) {
  return {
    title,
    xaxis: {
      title: {
        text: "x",
        standoff: 12
      },
      range: [-2.1, 2.1]
    },
    yaxis: {
      title: "y"
    },
    margin: {
      l: 55,
      r: 25,
      t: 50,
      b: 100
    },
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.28
    }
  };
}


function createPlotOptions() {
  return {
    responsive: true,

    // Die Plotly-Werkzeugleiste wird ausgeblendet, damit sie den Titel nicht überdeckt.
    displayModeBar: false
  };
}


function randomUniform(min, max, random) {
  return min + (max - min) * random();
}


function randomGaussian(mean, standardDeviation, random) {
  let u1 = 0;
  let u2 = 0;

  while (u1 === 0) {
    u1 = random();
  }

  while (u2 === 0) {
    u2 = random();
  }

  // Box-Muller-Transformation für normalverteiltes Rauschen.
  const z0 =
    Math.sqrt(-2.0 * Math.log(u1)) *
    Math.cos(2.0 * Math.PI * u2);

  return mean + z0 * standardDeviation;
}


function shuffleArray(array, random) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}


/*
  Seed-basierter Zufallsgenerator.
  Dadurch entstehen bei jedem Neuladen dieselben Datenpunkte.
  Das macht die Ergebnisse vergleichbarer und die Dokumentation stabiler.
*/
function mulberry32(seed) {
  return function () {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}