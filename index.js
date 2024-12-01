const videoElement = document.getElementById("webcam");
const btnStop = document.getElementById("stopButton");
const btnStart = document.getElementById("startButton");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
let mediaStream = null;
let captureInterval = null;
let capturedFrames = [];
let responses = [];

// Function to start webcam
async function startWebcam() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = mediaStream;
    console.log("Webcam started.");
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
}

// Function to stop webcam
function stopWebcam() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    videoElement.srcObject = null;
    clearInterval(captureInterval); // Stop capturing frames
    console.log("Webcam stopped.");
  }
}

// API calling function
const generate_emotion = async (url, imageDataUrl) => {
  const form = new FormData();
  form.append("image", imageDataUrl);
  try {
    const response = await fetch(url, {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error in API call:", error);
    return null;
  }
};

// Function to start capturing frames
function startCapturingFrames() {
  // Set canvas dimensions to match video dimensions
  canvas.width = videoElement.videoWidth / 2;
  canvas.height = videoElement.videoHeight / 2;

  // Capture frames every 4000 milliseconds (4 seconds)
  captureInterval = setInterval(async () => {
    // Ensure video is playing and dimensions are set
    if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL("image/jpeg", 0.5);
      capturedFrames.push(imageDataUrl); // Store the captured frame
      console.log(`Captured frame ${capturedFrames.length}:`, imageDataUrl);

      const url = "http://127.0.0.1:5000/analyze_emotion";
      const response = await generate_emotion(url, imageDataUrl);
      if (response) {
        responses.push(response);
      }
    }
  }, 4000); // Adjust the interval as needed (4000 ms = 4 seconds)
}

// Event listeners for buttons
btnStart.addEventListener("click", () => {
  startWebcam();
  videoElement.addEventListener("loadedmetadata", () => {
    // Start capturing frames when metadata is loaded
    startCapturingFrames();
  });
});

btnStop.addEventListener("click", () => {
  stopWebcam();
  console.log("Captured Frames:", capturedFrames);
  console.log("Responses:", responses);
});
