// Wrap all code in a DOMContentLoaded listener to ensure elements exist
document.addEventListener('DOMContentLoaded', (event) => {
    console.log("SCRIPT: DOM fully loaded and parsed.");

    // --- Global Scope Variables (Consider scoping them if possible) ---
    let pose = null;
    let standardPoseData = null;
    let animationFrameId = null;
    let isDetecting = false;
    // Define POSE_CONNECTIONS (Keep global if needed by multiple parts, otherwise move)
    const POSE_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
        [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
        [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23],
        [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29],
        [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
    ];
    const POSE_SOLUTION_OPTIONS = {
        modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false,
        minDetectionConfidence: 0.5, minTrackingConfidence: 0.5
    };
    const ANGLES_TO_COMPARE = {
        "left_elbow":   { p1: 11, p2: 13, p3: 15, threshold: 25 },
        "right_elbow":  { p1: 12, p2: 14, p3: 16, threshold: 25 },
        "left_shoulder":{ p1: 13, p2: 11, p3: 23, threshold: 20 },
        "right_shoulder":{ p1: 14, p2: 12, p3: 24, threshold: 20 },
        "left_hip":     { p1: 11, p2: 23, p3: 25, threshold: 20 },
        "right_hip":    { p1: 12, p2: 24, p3: 26, threshold: 20 },
        "left_knee":    { p1: 23, p2: 25, p3: 27, threshold: 30 },
        "right_knee":   { p1: 24, p2: 26, p3: 28, threshold: 30 },
    };

    // --- Helper Functions ---
    function getAngle(p1, p2, p3) {
        if (!p1 || !p2 || !p3 || p1.visibility < 0.4 || p2.visibility < 0.4 || p3.visibility < 0.4) return null;
        const angleRad = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
        let angleDeg = Math.abs(angleRad * 180.0 / Math.PI);
        if (angleDeg > 180.0) angleDeg = 360.0 - angleDeg;
        return angleDeg;
    }

    function calculateUserAngles(landmarks) {
        const angles = {};
        if (!landmarks) return angles;
        for (const angleName in ANGLES_TO_COMPARE) {
            const config = ANGLES_TO_COMPARE[angleName];
            angles[angleName + "_angle"] = getAngle(landmarks[config.p1], landmarks[config.p2], landmarks[config.p3]);
        }
        return angles;
    }

    function findClosestFrame(data, targetTime) {
        if (!data || data.length === 0) return null;
        let closest = data[0];
        let minDiff = Math.abs(data[0].timestamp - targetTime);
        for (let i = 1; i < data.length; i++) {
            const diff = Math.abs(data[i].timestamp - targetTime);
            if (diff < minDiff) { minDiff = diff; closest = data[i]; }
            if (data[i].timestamp > targetTime && diff > minDiff) break;
        }
        return closest;
    }

    // --- MediaPipe Pose Initialization (Only needed for Practice Page) ---
    function initializePose(callback) { // Added callback
        console.log("SCRIPT: Attempting to initialize MediaPipe Pose...");
        if (typeof Pose === 'undefined') {
            console.error("SCRIPT ERROR: MediaPipe Pose library not loaded!");
            return false;
        }
        try {
            pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`});
            pose.setOptions(POSE_SOLUTION_OPTIONS);
            pose.onResults(callback); // Use the provided callback
            console.log("SCRIPT: MediaPipe Pose initialized successfully.");
            return true;
        } catch (error) {
            console.error("SCRIPT ERROR: Error initializing MediaPipe Pose:", error);
            return false;
        }
    }

    // ==================================================
    // == PRACTICE PAGE LOGIC ==
    // ==================================================
    const practicePageContainer = document.querySelector('.practice-container');
    if (practicePageContainer) {
        console.log("SCRIPT: Practice page container FOUND. Setting up practice logic...");
        const webcamVideoElement = document.getElementById('webcamVideo');
        const standardVideoElement = document.getElementById('standardVideo');
        const canvasElement = document.getElementById('outputCanvas');
        const canvasCtx = canvasElement.getContext('2d');
        const feedbackElement = document.getElementById('feedback');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (!webcamVideoElement || !standardVideoElement || !canvasElement || !canvasCtx || !feedbackElement || !startBtn || !stopBtn) {
            console.error("SCRIPT ERROR: One or more required elements for the practice page are missing!");
        } else {
            console.log("SCRIPT: All practice page elements found.");

            // --- Practice Page Specific Functions ---
            async function loadStandardData() {
                console.log("SCRIPT (Practice): Attempting to load standard pose data...");
                feedbackElement.textContent = "Loading standard action data...";
                startBtn.disabled = true;
                try {
                    const response = await fetch('/get_standard_data'); // <<< THIS is the call causing the log
                    console.log("SCRIPT (Practice): Fetch response status:", response.status);
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    standardPoseData = await response.json();
                    if (!standardPoseData || !Array.isArray(standardPoseData) || standardPoseData.length === 0) {
                        throw new Error("Standard pose data is empty or invalid.");
                    }
                    console.log(`SCRIPT (Practice): Standard pose data loaded successfully (${standardPoseData.length} frames).`);
                    feedbackElement.textContent = "The data is loaded, please click Start Test.";
                    startBtn.disabled = false;
                } catch (error) {
                    console.error("SCRIPT ERROR (Practice): Error loading standard pose data:", error);
                    feedbackElement.textContent = `Error: Unable to load standard data (${error.message}).`; // Removed Chinese period
                    startBtn.disabled = true;
                }
            }

            async function startDetection() {
                console.log("SCRIPT (Practice): startDetection called.");
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    feedbackElement.textContent = "Error: Browser does not support camera access!"; return;
                }
                if (!pose) { feedbackElement.textContent = "Error: Pose model not initialized!"; return; } // Changed "Attitude" to "Pose" and "initialised" to "initialized"
                if (!standardPoseData) { feedbackElement.textContent = "Error: Standard data not loaded!"; return; }

                isDetecting = true;
                startBtn.disabled = true; stopBtn.disabled = false;
                feedbackElement.textContent = "Requesting camera permission..."; // Translated
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    feedbackElement.textContent = "Starting camera..."; // Translated
                    webcamVideoElement.srcObject = stream;
                    await new Promise((resolve, reject) => {
                        webcamVideoElement.onloadedmetadata = resolve;
                        webcamVideoElement.onerror = reject;
                    });
                    canvasElement.width = webcamVideoElement.videoWidth;
                    canvasElement.height = webcamVideoElement.videoHeight;
                    feedbackElement.textContent = "Detecting... Please follow the standard video."; // Translated
                    standardVideoElement.currentTime = 0;
                    standardVideoElement.play();
                    predictionLoop();
                } catch (err) {
                    console.error("SCRIPT ERROR (Practice): Error accessing webcam:", err);
                    feedbackElement.textContent = `Cannot access camera: ${err.message}.`; // Translated
                    stopDetection();
                }
            }

            function stopDetection() {
                console.log("SCRIPT (Practice): stopDetection called.");
                isDetecting = false;
                startBtn.disabled = false; stopBtn.disabled = true;
                standardVideoElement.pause();
                if (webcamVideoElement.srcObject) {
                    webcamVideoElement.srcObject.getTracks().forEach(track => track.stop());
                    webcamVideoElement.srcObject = null;
                }
                if (animationFrameId) cancelAnimationFrame(animationFrameId); animationFrameId = null;
                setTimeout(() => { if (!isDetecting && canvasCtx) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height); }, 100);
                feedbackElement.textContent = "Detection stopped."; // Translated
            }

            async function predictionLoop() {
                if (!isDetecting || !webcamVideoElement || webcamVideoElement.paused || webcamVideoElement.ended) {
                    if (isDetecting) stopDetection(); return;
                }
                if (pose && webcamVideoElement.readyState >= 2) {
                    try { await pose.send({ image: webcamVideoElement }); }
                    catch (error) { console.error("SCRIPT ERROR (Practice): Error sending frame:", error); stopDetection(); return; }
                }
                if (isDetecting) animationFrameId = requestAnimationFrame(predictionLoop);
            }

            function onPoseResults(results) { // This is the callback for initializePose
                if (!isDetecting || !canvasCtx) return;
                canvasCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                if (results.poseLandmarks) {
                    if (window.drawConnectors && window.drawLandmarks && POSE_CONNECTIONS) {
                        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
                        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', radius: 4 });
                    }
                    comparePose(results.poseLandmarks);
                }
                canvasCtx.restore();
            }

            function comparePose(currentUserLandmarks) {
                if (!standardPoseData || !standardVideoElement) return;
                const currentTime = standardVideoElement.currentTime;
                let closestFrame = findClosestFrame(standardPoseData, currentTime);
                if (!closestFrame || !closestFrame.angles) { feedbackElement.textContent = "Synchronizing standard data..."; return; } // Translated

                const currentUserAngles = calculateUserAngles(currentUserLandmarks);
                let feedbackMessages = []; let score = 100; const penalty = 5;
                for (const angleName in ANGLES_TO_COMPARE) {
                    const config = ANGLES_TO_COMPARE[angleName];
                    const userAngle = currentUserAngles[angleName + "_angle"];
                    const standardAngle = closestFrame.angles[angleName + "_angle"];
                    const threshold = config.threshold;
                    if (standardAngle != null) {
                        if (userAngle != null) {
                            const diff = Math.abs(userAngle - standardAngle);
                            if (diff > threshold) {
                                let part = angleName.replace('_', ' ');
                                let hint = userAngle < standardAngle ? "Open a bit more" : "Pull back a bit"; // Translated
                                feedbackMessages.push(`Your ${part} angle deviation (${userAngle.toFixed(0)}° vs ${standardAngle.toFixed(0)}°). ${hint}`); // Translated "您的"
                                score -= penalty;
                            }
                        } else { feedbackMessages.push(`Cannot detect your ${angleName.replace('_', ' ')}`); score -= penalty; } // Translated "无法检测到您的"
                    }
                }
                score = Math.max(0, score);
                if (feedbackMessages.length > 0) { feedbackElement.innerHTML = `Score: ${score}<br>${feedbackMessages.join('<br>')}`; } // Translated "得分"
                else { feedbackElement.textContent = `Action standard! Score: ${score}`; } // Translated "动作标准！得分"
            }

            // --- Event Listeners & Initialization for Practice Page ---
            console.log("SCRIPT (Practice): Adding event listeners.");
            startBtn.addEventListener('click', startDetection);
            stopBtn.addEventListener('click', stopDetection);

            console.log("SCRIPT (Practice): Running initialization.");
            // Initialize Pose and THEN load data
            if (initializePose(onPoseResults)) { // Pass the results handler as callback
                loadStandardData();
            } else {
                startBtn.disabled = true;
                feedbackElement.textContent = "Error: Pose model initialization failed."; // Translated
            }
        }
    } else {
        console.log("SCRIPT: Practice page container NOT FOUND. Skipping practice logic.");
    }

    // ==================================================
    // == UPLOAD PAGE LOGIC ==
    // ==================================================
    const uploadPageContainer = document.querySelector('.upload-container');
    if (uploadPageContainer) {
        console.log("SCRIPT: Upload page container FOUND. Setting up upload logic...");
        const videoUploadInput = document.getElementById('videoUpload');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadStatusElement = document.getElementById('uploadStatus');
        const fileLabel = uploadPageContainer.querySelector('.file-label'); // Scope label to this container

        if (!videoUploadInput || !uploadBtn || !uploadStatusElement || !fileLabel) {
            console.error("SCRIPT ERROR: One or more required elements for the upload page are missing!");
        } else {
            console.log("SCRIPT: All upload page elements found.");

            videoUploadInput.addEventListener('change', () => {
                if (videoUploadInput.files.length > 0) {
                    fileLabel.textContent = videoUploadInput.files[0].name;
                    uploadBtn.disabled = false; uploadStatusElement.textContent = '';
                } else {
                    fileLabel.textContent = "Select video file"; uploadBtn.disabled = true; // Translated
                }
            });
            uploadBtn.disabled = true; // Initially disable

            async function uploadVideo() {
                console.log("SCRIPT (Upload): uploadVideo called.");
                const file = videoUploadInput.files[0];
                if (!file) { uploadStatusElement.textContent = "Please select a video file first."; uploadStatusElement.style.color = 'red'; return; } // Translated

                // Basic client-side validation (optional but good)
                // ... (add type/size checks if needed) ...

                uploadStatusElement.textContent = "Uploading video..."; uploadStatusElement.style.color = 'var(--primary-color)'; // Translated
                uploadBtn.disabled = true; fileLabel.style.opacity = '0.6';
                const formData = new FormData(); formData.append('video', file);

                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const result = await response.json();
                    if (response.ok) {
                        uploadStatusElement.textContent = `Upload successful: ${result.message || 'Video received.'}`; uploadStatusElement.style.color = 'green'; // Translated
                        videoUploadInput.value = ''; fileLabel.textContent = "Select video file"; uploadBtn.disabled = true; // Translated
                    } else {
                        uploadStatusElement.textContent = `Upload failed: ${result.error || 'Server error.'}`; uploadStatusElement.style.color = 'red'; // Translated
                        uploadBtn.disabled = false; // Allow retry
                    }
                } catch (error) {
                    console.error("SCRIPT ERROR (Upload): Fetch error:", error);
                    uploadStatusElement.textContent = "Upload error, please check network."; uploadStatusElement.style.color = 'red'; // Translated
                    uploadBtn.disabled = false; // Allow retry
                } finally {
                    fileLabel.style.opacity = '1';
                    console.log("SCRIPT (Upload): Upload process finished.");
                }
            }
            uploadBtn.addEventListener('click', uploadVideo);
        }
    } else {
        console.log("SCRIPT: Upload page container NOT FOUND. Skipping upload logic.");
    }

    // ==================================================
    // == HERB IDENTIFIER PAGE LOGIC ==
    // ==================================================
    const herbPageContainer = document.querySelector('.herb-identifier-container');
    if (herbPageContainer) {
        console.log("SCRIPT: Herb identifier page container FOUND. Setting up logic..."); // Log 2
        const imageUploadInput = document.getElementById('herbImageUpload');
        const predictBtn = document.getElementById('predictBtn');
        const predictionResultElement = document.getElementById('predictionResult');
        const imagePreview = document.getElementById('imagePreview');
        const fileLabel = herbPageContainer.querySelector('.file-label'); // Scope label to this container

        if (!imageUploadInput || !predictBtn || !predictionResultElement || !imagePreview || !fileLabel) {
            console.error("SCRIPT ERROR: One or more required elements for the herb identifier page are missing!");
        } else {
            console.log("SCRIPT: All herb identifier page elements found.");

            // --- Image Preview Logic ---
            imageUploadInput.addEventListener('change', function() {
                console.log("SCRIPT (Herb): Image input changed."); // Log 3
                const file = this.files[0];
                if (file) {
                    if (!file.type.startsWith('image/')){
                         predictionResultElement.textContent = "Please select an image file."; predictionResultElement.style.color = 'red'; // Translated
                         imagePreview.style.display = 'none'; imagePreview.src = '#';
                         fileLabel.textContent = "Select image file"; predictBtn.disabled = true; // Translated
                         return;
                    }
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                        console.log("SCRIPT (Herb): Image preview updated."); // Log 4
                    }
                    reader.readAsDataURL(file);
                    fileLabel.textContent = file.name;
                    predictBtn.disabled = false; // Enable button
                    predictionResultElement.textContent = ''; predictionResultElement.style.color = 'inherit';
                    console.log("SCRIPT (Herb): Predict button enabled."); // Log 5
                } else {
                    imagePreview.style.display = 'none'; imagePreview.src = '#';
                    fileLabel.textContent = "Select image file"; // Translated
                    predictBtn.disabled = true; // Disable button
                    console.log("SCRIPT (Herb): Predict button disabled (no file)."); // Log 6
                }
            });

            // --- Prediction Logic ---
            async function predictHerb() {
                console.log("SCRIPT (Herb): predictHerb function CALLED."); // Log 7
                const file = imageUploadInput.files[0];
                if (!file) {
                    predictionResultElement.textContent = "Please select an image first."; predictionResultElement.style.color = 'red'; // Translated
                    return;
                }

                predictionResultElement.textContent = "Identifying, please wait..."; predictionResultElement.style.color = 'var(--primary-color)'; // Translated
                predictBtn.disabled = true; fileLabel.style.opacity = '0.6';
                console.log("SCRIPT (Herb): Starting herb prediction fetch...");

                const formData = new FormData(); formData.append('image', file);

                try {
                    const response = await fetch('/predict_herb', { method: 'POST', body: formData }); // <<< THIS is the expected fetch
                    console.log("SCRIPT (Herb): Prediction fetch response status:", response.status);
                    const result = await response.json();
                    console.log("SCRIPT (Herb): Prediction response data:", result);
                    if (response.ok) {
                        predictionResultElement.innerHTML = `Identification result: <strong>${result.prediction}</strong><br>(Confidence: ${result.confidence})`; // Translated
                        predictionResultElement.style.color = 'green';
                    } else {
                        predictionResultElement.textContent = `Identification failed: ${result.error || `Server error (${response.status})`}`; // Translated
                        predictionResultElement.style.color = 'red';
                    }
                } catch (error) {
                    console.error("SCRIPT ERROR (Herb): Fetch error:", error);
                    predictionResultElement.textContent = "Identification request error, please check network."; // Translated
                    predictionResultElement.style.color = 'red';
                } finally {
                    predictBtn.disabled = false; // Re-enable button
                    fileLabel.style.opacity = '1';
                    console.log("SCRIPT (Herb): Herb prediction process finished.");
                }
            }

            // --- Event Listener ---
            if (predictBtn) {
                 predictBtn.addEventListener('click', predictHerb);
                 console.log("SCRIPT (Herb): Event listener added to predict button."); // Log 8
            } else {
                 console.error("SCRIPT ERROR (Herb): Predict button not found!");
            }
        }
    } else {
        console.log("SCRIPT: Herb identifier page container NOT FOUND. Skipping logic setup."); // Log 9
    }

    console.log("SCRIPT: End of DOMContentLoaded execution.");
}); // End of DOMContentLoaded listener