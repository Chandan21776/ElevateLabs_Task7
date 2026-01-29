let data = [];
let headers = [];

document.getElementById("fileInput").addEventListener("change", function (e) {
  const reader = new FileReader();
  reader.onload = () => parseCSV(reader.result);
  reader.readAsText(e.target.files[0]);
});

function parseCSV(text) {
  const rows = text.trim().split("\n");
  headers = rows[0].split(",");
  data = rows.slice(1).map(r => r.split(","));
}

function runModel() {
  let processed = preprocessData();
  let X = processed.X;
  let y = processed.y;

  const split = Math.floor(0.7 * X.length);
  const X_train = X.slice(0, split);
  const y_train = y.slice(0, split);
  const X_test = X.slice(split);
  const y_test = y.slice(split);

  let weights = new Array(X_train[0].length).fill(0);

  for (let epoch = 0; epoch < 800; epoch++) {
    X_train.forEach((x, i) => {
      let z = dot(x, weights);
      let pred = sigmoid(z);
      let error = y_train[i] - pred;
      x.forEach((val, j) => weights[j] += 0.01 * error * val);
    });
  }

  let probs = X_test.map(x => sigmoid(dot(x, weights)));
  let preds = probs.map(p => p >= 0.5 ? 1 : 0);

  evaluate(y_test, preds, probs);
}

/* ---------- Preprocessing ---------- */
function preprocessData() {
  let X = [];
  let y = [];

  data.forEach(r => {
    let age = r[5] === "" ? 30 : +r[5];
    let fare = r[9] === "" ? 0 : +r[9];
    let sex = r[4] === "male" ? 1 : 0;

    X.push([sex, age / 80, fare / 500]);
    y.push(+r[1]);
  });

  return { X, y };
}

/* ---------- Evaluation ---------- */
function evaluate(actual, predicted, probs) {
  let tp = 0, tn = 0, fp = 0, fn = 0;

  predicted.forEach((p, i) => {
    if (p === 1 && actual[i] === 1) tp++;
    else if (p === 0 && actual[i] === 0) tn++;
    else if (p === 1 && actual[i] === 0) fp++;
    else fn++;
  });

  let accuracy = (tp + tn) / actual.length;
  let precision = tp / (tp + fp);
  let recall = tp / (tp + fn);
  let f1 = 2 * (precision * recall) / (precision + recall);

  document.getElementById("accuracy").innerText = "Accuracy: " + accuracy.toFixed(2);
  document.getElementById("precision").innerText = "Precision: " + precision.toFixed(2);
  document.getElementById("recall").innerText = "Recall: " + recall.toFixed(2);
  document.getElementById("f1").innerText = "F1-Score: " + f1.toFixed(2);

  document.getElementById("tp").innerText = tp;
  document.getElementById("tn").innerText = tn;
  document.getElementById("fp").innerText = fp;
  document.getElementById("fn").innerText = fn;

  drawROC(actual, probs);
}

/* ---------- ROC & AUC ---------- */
function drawROC(actual, probs) {
  const ctx = document.getElementById("roc").getContext("2d");
  ctx.clearRect(0, 0, 500, 300);

  let points = [];
  for (let t = 0; t <= 1; t += 0.05) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    probs.forEach((p, i) => {
      let pred = p >= t ? 1 : 0;
      if (pred === 1 && actual[i] === 1) tp++;
      else if (pred === 0 && actual[i] === 0) tn++;
      else if (pred === 1 && actual[i] === 0) fp++;
      else fn++;
    });
    points.push([fp / (fp + tn), tp / (tp + fn)]);
  }

  ctx.beginPath();
  ctx.moveTo(0, 300);
  points.forEach(p => ctx.lineTo(p[0] * 500, 300 - p[1] * 300));
  ctx.strokeStyle = "#e74c3c";
  ctx.stroke();

  let auc = points.reduce((a, p, i) => {
    if (i === 0) return 0;
    return a + (points[i][0] - points[i - 1][0]) * points[i][1];
  }, 0);

  document.getElementById("auc").innerText = "AUC Score: " + auc.toFixed(2);
}

/* ---------- Utils ---------- */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
