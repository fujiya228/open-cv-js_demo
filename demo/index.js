/**
 * メモリに気を付けたい
 * 以下のやつ参考にしたら1541回目に確実に死を迎えた笑（chrome）
 * https://blog.obniz.io/make/opencv-drawing-camera/
 */
const FPS = 30;
const GS = 'GrayScale';
const LD = 'LineDrawing';
const FD = 'FaceDetection';

const utils = new Utils('error-message');

let isStreaming = false;
let type;
const video = document.getElementById('video');
const switchGrayScale = document.getElementById('switchGrayScale');
const switchLineDrawing = document.getElementById('switchLineDrawing');
const switchFaceDetection = document.getElementById('switchFaceDetection');
const canvasOutput = document.getElementById('canvasOutput');
const canvasContext = canvasOutput.getContext('2d');
const count = 0;
// 共通
let cap;
let src;
let dst;
let gray;
// 線画用
let imgDilated;
let imgDiff;
// 顔検出用
let faces;
let classifier;
// 計測用
let begin;
let loss;
let delay;

const faceCascadeFile = 'haarcascade_frontalface_default.xml';
utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
  switchGrayScale.removeAttribute('disabled');
  switchLineDrawing.removeAttribute('disabled');
  switchFaceDetection.removeAttribute('disabled');
});
// イベントリスナの設定
switchGrayScale.addEventListener('click', { typeText: GS, handleEvent: clickSwitch });
switchLineDrawing.addEventListener('click', { typeText: LD, handleEvent: clickSwitch });
switchFaceDetection.addEventListener('click', { typeText: FD, handleEvent: clickSwitch });

// 切り替え
function clickSwitch() {
  if (isStreaming) {
    // 動いていたら一旦停止
    onVideoStopped();
    // 動いていたもののボタンならそのまま終了
    if (type === this.typeText) {
      return;
    }
  }

  // 違うボタンなら再始動
  type = this.typeText;
  utils.clearError();
  utils.startCamera('env', onVideoStarted, 'video');
}

// 前処理
function onVideoStarted(stream, self_video) {
  // console.log(stream) // utils.jsにあったから確認で
  // console.log(self_video) // utils.jsにあったから確認で
  isStreaming = true;

  video.width = video.videoWidth;
  video.height = video.videoHeight;
  cap = new cv.VideoCapture(video);
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  gray = new cv.Mat();
  switch (type) {
    case GS:
      // グレースケール
      switchGrayScale.innerText = 'Stop';
      switchGrayScale.classList.add('on');
      startDrawing(grayScale);
      break;
    case LD:
      // 線画
      switchLineDrawing.innerText = 'Stop';
      switchLineDrawing.classList.add('on');
      imgDilated = new cv.Mat();
      imgDiff = new cv.Mat();
      startDrawing(convertImageToLineDrawing);
      break;
    case FD:
      // 顔検出
      switchFaceDetection.innerText = 'Stop';
      switchFaceDetection.classList.add('on');
      faces = new cv.RectVector();
      classifier = new cv.CascadeClassifier();
      classifier.load('haarcascade_frontalface_default.xml');
      // console.log('model load ' + classifier.load('haarcascade_frontalface_default.xml'));
      startDrawing(faceDetection);
      break;
    default:
      console.log(type);
  }
}

// 後処理
function onVideoStopped() {
  // 描画していたものをクリア
  canvasContext.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
  // カメラ停止
  utils.stopCamera();
  isStreaming = false;
  // destructorがないらしいので手動で削除
  src.delete();
  dst.delete();
  gray.delete();
  switch (type) {
    case GS:
      // グレースケール
      switchGrayScale.innerText = type;
      switchGrayScale.classList.remove('on');
      break;
    case LD:
      // 線画
      switchLineDrawing.innerText = type;
      switchLineDrawing.classList.remove('on');
      imgDiff.delete();
      imgDilated.delete();
      break;
    case FD:
      // 顔検出
      switchFaceDetection.innerText = type;
      switchFaceDetection.classList.remove('on');
      faces.delete();
      classifier.delete();
      break;
    default:
      console.log(type);
  }
}

// 実行
function startDrawing(callBack) {
  if (!isStreaming) {
    return;
  }

  begin = Date.now(); // 開始
  /* ====================================================== */
  cap.read(src); // 読み込み

  // 処理の呼び出し
  callBack();

  cv.imshow('canvasOutput', dst); // 出力
  /* ====================================================== */
  loss = Date.now() - begin; // 計算時間
  delay = (1000 / FPS) - loss; // 遅延計算
  setTimeout(startDrawing, delay, callBack); // 再帰
  // console.log(count++ + ' 処理時間：' + loss + ' ms') // 確認
}

// 処理の関数

// グレースケール
function grayScale() {
  cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
}

// 線画
function convertImageToLineDrawing() {
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  cv.dilate(gray, imgDilated, kernel, new cv.Point(-1, 1), 1);

  cv.absdiff(imgDilated, gray, imgDiff);

  cv.bitwise_not(imgDiff, dst);
}

// 顔検出
function faceDetection() {
  src.copyTo(dst);
  cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
  classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
  for (let i = 0; i < faces.size(); ++i) {
    const face = faces.get(i);
    const point1 = new cv.Point(face.x, face.y);
    // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
    const point2 = new cv.Point(face.x + face.width, face.y + face.height);
    cv.rectangle(dst, point1, point2, [103, 183, 179, 255]);
  }
}
