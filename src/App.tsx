import { useState, useRef, useEffect } from 'react';
import { Eraser, RefreshCw } from 'lucide-react';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [predictions, setPredictions] = useState<Array<{ digit: number; probability: number }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 25;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let x, y;
    if ('touches' in e) {
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    }

    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let x, y;
    if ('touches' in e) {
      e.preventDefault();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    predictDigit();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(0);
    setPredictions([]);
  };

  const predictDigit = async () => {
  const canvas = canvasRef.current!;
  const dataUrl = canvas.toDataURL("image/png");
  const res = await fetch("https://handwritten-digit-recognizer-gtgb.onrender.com/predict/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: dataUrl }),
  });

  const result = await res.json();
  if(result.prediction === -1){
    setPrediction(null);
    setPredictions([]);
    return;
  }
  setPrediction(result.prediction);
  const probs = result.probabilities.map((p: number, i: number) => ({
    digit: i,
    probability: p,
  }));

  setPredictions(probs.sort((a, b) => b.probability - a.probability));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Digits recognizer
          </h1>
          <p className="text-gray-600">
            Draw a number from 0 to 9 and the model will recognize it automatically
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Panel de dibujo */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-2">
                Draw here!
              </h2>
              <p className="text-sm text-gray-600">
                Use your mouse or finger to draw a digit.
              </p>
            </div>

            <div className="relative">
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
                className="border-4 border-gray-300 rounded-lg cursor-crosshair w-full touch-none"
                style={{ maxWidth: '400px', aspectRatio: '1/1' }}
              />
            </div>

            <div className="mt-4 flex gap-3">
              <button
                onClick={clearCanvas}
                className="flex-1 bg-red-500 hover:bg-red-600 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
              >
                <Eraser className="w-5 h-5" />
                Clear
              </button>
              <button
                onClick={predictDigit}
                className="flex-1 bg-indigo-500 hover:bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Predict
              </button>
            </div>
          </div>

          {/* Panel de predicción */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-2">
                Prediction
              </h2>
              <p className="text-sm text-gray-600">
                Machine Learning Model Output
              </p>
            </div>

            {prediction !== null ? (
              <div className="space-y-6">
                {/* Predicción principal */}
                <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl p-8 text-center text-white">
                  <p className="text-lg mb-2 opacity-90">Digit detected</p>
                  <p className="text-8xl font-bold mb-2">{prediction}</p>
                </div>

                {/* Todas las probabilidades */}
                <div>
                  <h3 className="font-semibold text-gray-700 mb-3">
                    Digit probabilities
                  </h3>
                  <div className="space-y-2">
                    {predictions.map((pred) => (
                      <div key={pred.digit} className="flex items-center gap-3">
                        <span className="w-8 font-semibold text-gray-700">
                          {pred.digit}
                        </span>
                        <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              pred.digit === prediction
                                ? 'bg-gradient-to-r from-indigo-500 to-purple-600'
                                : 'bg-gray-400'
                            }`}
                            style={{ width: `${pred.probability * 100}%` }}
                          />
                        </div>
                        <span className="w-16 text-sm text-gray-600 text-right">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-gray-400">
                <svg
                  className="w-24 h-24 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                  />
                </svg>
                <p className="text-lg font-medium">Draw a digit</p>
                <p className="text-sm">The model will recognize it automatically</p>
              </div>
            )}
          </div>
        </div>

        {/* Información adicional */}
        <div className="mt-8 bg-white rounded-2xl shadow-xl p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            About this model
          </h3>
          <p className="text-gray-600 mb-3">
              This is a Machine Learning project that performs handwritten digit recognition.
              In a real-world implementation, a convolutional neural network (CNN) trained on
              the MNIST dataset would be used, which contains 60,000 handwritten digit training
              images.
          </p>
          <div className="grid sm:grid-cols-3 gap-4 mt-4">
            <div className="bg-indigo-50 rounded-lg p-4">
              <p className="text-sm font-semibold text-indigo-900 mb-1">
                Technology
              </p>
              <p className="text-xs text-indigo-700">
                FrontEnd: React + Canvas API
                BackEnd: Python+ Numpy
              </p>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <p className="text-sm font-semibold text-purple-900 mb-1">
                Model
              </p>
              <p className="text-xs text-purple-700">
                Model built with NumPy and trained on the MNIST dataset
              </p>
            </div>
            <div className="bg-pink-50 rounded-lg p-4">
              <p className="text-sm font-semibold text-pink-900 mb-1">
                Accuracy
              </p>
              <p className="text-xs text-pink-700">
                Experimental model – still improving
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
