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

            // --- MODIFIED startDetection ---
            async function startDetection() {
                console.log("SCRIPT (Practice): startDetection called.");
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    feedbackElement.textContent = "Error: Browser does not support camera access!"; return;
                }
                if (!pose) { feedbackElement.textContent = "Error: Pose model not initialized!"; return; }
                if (!standardPoseData) { feedbackElement.textContent = "Error: Standard data not loaded!"; return; }

                isDetecting = true; // Set flag early
                startBtn.disabled = true;
                stopBtn.disabled = false;
                feedbackElement.textContent = "Requesting camera permission...";

                let stream = null; // Declare stream outside try block

                try {
                    // 1. Get camera stream
                    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    feedbackElement.textContent = "Starting camera...";
                    webcamVideoElement.srcObject = stream;

                    // Wait for metadata to load
                    await new Promise((resolve, reject) => {
                        webcamVideoElement.onloadedmetadata = resolve;
                        webcamVideoElement.onerror = (e) => reject(new Error("Webcam metadata loading failed."));
                    });

                    canvasElement.width = webcamVideoElement.videoWidth;
                    canvasElement.height = webcamVideoElement.videoHeight;
                    feedbackElement.textContent = "Detecting... Please follow the standard video.";

                    // 2. Start standard video playback, handling potential interruption
                    standardVideoElement.currentTime = 0;
                    try {
                        await standardVideoElement.play(); // Wait for play() promise
                        console.log("SCRIPT (Practice): Standard video playback started.");
                    } catch (playError) {
                        // Ignore AbortError ONLY if detection is already stopping/stopped
                        if (playError.name === 'AbortError' && !isDetecting) {
                            console.warn("SCRIPT (Practice): Standard video play() aborted, likely due to immediate stopDetection call. Ignoring.");
                        } else {
                            // Re-throw other playback errors
                            console.error("SCRIPT ERROR (Practice): Error playing standard video:", playError);
                            throw playError; // Propagate error to outer catch block
                        }
                    }

                    // 3. Start the prediction loop ONLY if still detecting
                    if (isDetecting) {
                        predictionLoop();
                    } else {
                         console.warn("SCRIPT (Practice): Detection stopped before prediction loop could start.");
                         // Ensure cleanup if stopped very quickly
                         if (stream) stream.getTracks().forEach(track => track.stop());
                         webcamVideoElement.srcObject = null;
                    }

                } catch (err) {
                    console.error("SCRIPT ERROR (Practice): Error during startDetection setup:", err);
                    feedbackElement.textContent = `Cannot start detection: ${err.message}.`;
                    // Ensure cleanup happens even if setup fails
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                    if (webcamVideoElement) webcamVideoElement.srcObject = null; // Clear srcObject if element exists
                    stopDetection(); // Call stopDetection to reset state properly
                }
            }
            // --- END MODIFIED startDetection ---

            // --- MODIFIED stopDetection ---
            function stopDetection() {
                console.log("SCRIPT (Practice): stopDetection called.");
                const wasDetecting = isDetecting; // Store previous state
                isDetecting = false; // Set flag immediately
                startBtn.disabled = false;
                stopBtn.disabled = true;

                // Pause standard video only if it exists and isn't already paused
                if (standardVideoElement && !standardVideoElement.paused) {
                    standardVideoElement.pause();
                    console.log("SCRIPT (Practice): Standard video paused.");
                }

                // Stop webcam stream if it exists
                if (webcamVideoElement && webcamVideoElement.srcObject) {
                    webcamVideoElement.srcObject.getTracks().forEach(track => track.stop());
                    webcamVideoElement.srcObject = null;
                    console.log("SCRIPT (Practice): Webcam stream stopped.");
                }

                // Cancel animation frame
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                    console.log("SCRIPT (Practice): Animation frame cancelled.");
                }

                // Clear canvas after a short delay, only if detection was active before stopping
                setTimeout(() => {
                    // Check isDetecting *again* inside timeout to prevent clearing if detection restarts quickly
                    if (!isDetecting && canvasCtx) {
                         canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                         console.log("SCRIPT (Practice): Canvas cleared.");
                    }
                }, 100);

                // Update feedback only if detection was active before stopping
                if (wasDetecting) {
                    feedbackElement.textContent = "Detection stopped."; // Translated
                }
            }
            // --- END MODIFIED stopDetection ---


            async function predictionLoop() {
                if (!isDetecting || !webcamVideoElement || webcamVideoElement.paused || webcamVideoElement.ended) {
                    if (isDetecting) stopDetection();
                    return;
                }
                if (pose && webcamVideoElement.readyState >= 2) {
                    try {
                        await pose.send({ image: webcamVideoElement });
                    }
                    catch (error) {
                        console.error("SCRIPT ERROR (Practice): Error sending frame to MediaPipe:", error);
                        stopDetection();
                        return;
                    }
                }
                if (isDetecting) {
                    animationFrameId = requestAnimationFrame(predictionLoop);
                }
            }


            function onPoseResults(results) {
                if (!isDetecting || !canvasCtx) return;

                canvasCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

                if (results.poseLandmarks) {
                    const landmarks = results.poseLandmarks;
                    comparePose(landmarks);
                    const BODY_KEYPOINTS = [11,12,13,14,15,16,23,24,25,26,27,28];
                    const wrongSet = window.wrongAnglePoints || new Set();

                    for (const i of BODY_KEYPOINTS) {
                        const kp = landmarks[i];
                        if (!kp || kp.visibility < 0.4) continue;

                        const x = kp.x * canvasElement.width;
                        const y = kp.y * canvasElement.height;
                        const color = wrongSet.has(i) ? "#FF0000" : "#00FF00";

                        canvasCtx.beginPath();
                        canvasCtx.arc(x, y, 6, 0, 2 * Math.PI);
                        canvasCtx.fillStyle = color;
                        canvasCtx.fill();
                    }

                    if (window.drawConnectors && POSE_CONNECTIONS) {
                        drawConnectors(canvasCtx, landmarks, POSE_CONNECTIONS.filter(pair =>
                            BODY_KEYPOINTS.includes(pair[0]) && BODY_KEYPOINTS.includes(pair[1])
                        ), { color: '#AAAAAA', lineWidth: 2 });
                    }
                }
                canvasCtx.restore();
            }

            function comparePose(currentUserLandmarks) {
                if (!standardPoseData || !standardVideoElement) return;
                const currentTime = standardVideoElement.currentTime;
                let closestFrame = findClosestFrame(standardPoseData, currentTime);
                if (!closestFrame || !closestFrame.angles || !closestFrame.landmarks) {
                    feedbackElement.textContent = "Synchronizing standard data...";
                    return;
                }

                const currentUserAngles = calculateUserAngles(currentUserLandmarks);
                let feedbackMessages = [];
                let score = 100;
                const penalty = 5;
                window.wrongAnglePoints = new Set();

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
                                let hint = userAngle < standardAngle ? "Open a bit more" : "Pull back a bit";
                                feedbackMessages.push(`Your ${part} angle deviation (${userAngle.toFixed(0)}° vs ${standardAngle.toFixed(0)}°). ${hint}`);
                                score -= penalty;
                                window.wrongAnglePoints.add(config.p1);
                                window.wrongAnglePoints.add(config.p2);
                                window.wrongAnglePoints.add(config.p3);
                            }
                        } else {
                            feedbackMessages.push(`Cannot detect your ${angleName.replace('_', ' ')}`);
                            score -= penalty;
                        }
                    }
                }

                const relativePositionRules = [
                    { userIdx1: 15, userIdx2: 11, stdIdx1: 15, stdIdx2: 11, axis: 'y', part: 'left hand height' },
                    { userIdx1: 16, userIdx2: 12, stdIdx1: 16, stdIdx2: 12, axis: 'y', part: 'right hand height' },
                    { userIdx1: 25, userIdx2: 23, stdIdx1: 25, stdIdx2: 23, axis: 'y', part: 'left knee bend' },
                    { userIdx1: 26, userIdx2: 24, stdIdx1: 26, stdIdx2: 24, axis: 'y', part: 'right knee bend' },
                    { userIdx1: 27, userIdx2: 28, stdIdx1: 27, stdIdx2: 28, axis: 'x', part: 'feet distance' },
                    { userIdx1: 11, userIdx2: 23, stdIdx1: 11, stdIdx2: 23, axis: 'x', part: 'left trunk alignment' },
                    { userIdx1: 12, userIdx2: 24, stdIdx1: 12, stdIdx2: 24, axis: 'x', part: 'right trunk alignment' },
                    { userIdx1: 13, userIdx2: 11, stdIdx1: 13, stdIdx2: 11, axis: 'y', part: 'left elbow height' },
                    { userIdx1: 14, userIdx2: 12, stdIdx1: 14, stdIdx2: 12, axis: 'y', part: 'right elbow height' }
                ];

                const torsoLenUser = Math.abs(currentUserLandmarks[11]?.y - currentUserLandmarks[23]?.y || 0.3);
                const torsoLenStd = Math.abs(closestFrame.landmarks[11]?.y - closestFrame.landmarks[23]?.y || 0.3);

                for (const rule of relativePositionRules) {
                    const u1 = currentUserLandmarks[rule.userIdx1];
                    const u2 = currentUserLandmarks[rule.userIdx2];
                    const s1 = closestFrame.landmarks[rule.stdIdx1];
                    const s2 = closestFrame.landmarks[rule.stdIdx2];

                    if (!u1 || !u2 || !s1 || !s2 || u1.visibility < 0.4 || u2.visibility < 0.4 || s1.visibility < 0.4 || s2.visibility < 0.4) continue;

                    const userDiff = (u1[rule.axis] - u2[rule.axis]) / torsoLenUser;
                    const stdDiff = (s1[rule.axis] - s2[rule.axis]) / torsoLenStd;
                    const deviation = Math.abs(userDiff - stdDiff);

                    const directionWrong = (stdDiff > 0 && userDiff < 0) || (stdDiff < 0 && userDiff > 0);
                    const autoTolerance = Math.max(Math.abs(stdDiff), 0.15) * 0.5;

                    if (deviation > autoTolerance && directionWrong) {
                        feedbackMessages.push(`Your ${rule.part} is incorrect in relative ${rule.axis}-position.`);
                        score -= penalty;
                        window.wrongAnglePoints.add(rule.userIdx1);
                        window.wrongAnglePoints.add(rule.userIdx2);
                    }
                }

                score = Math.max(0, score);
                if (feedbackMessages.length > 0) {
                    feedbackElement.innerHTML = `Score: ${score}<br>${feedbackMessages.join('<br>')}`;
                } else {
                    feedbackElement.textContent = `Action standard! Score: ${score}`;
                }
            }

            // --- Event Listeners & Initialization for Practice Page ---
            console.log("SCRIPT (Practice): Adding event listeners.");
            startBtn.addEventListener('click', startDetection);
            stopBtn.addEventListener('click', stopDetection);

            console.log("SCRIPT (Practice): Running initialization.");
            if (initializePose(onPoseResults)) {
                loadStandardData();
            } else {
                startBtn.disabled = true;
                feedbackElement.textContent = "Error: Pose model initialization failed.";
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
        const fileLabel = uploadPageContainer.querySelector('.file-label');

        if (!videoUploadInput || !uploadBtn || !uploadStatusElement || !fileLabel) {
            console.error("SCRIPT ERROR: One or more required elements for the upload page are missing!");
        } else {
            console.log("SCRIPT: All upload page elements found.");

            videoUploadInput.addEventListener('change', () => {
                if (videoUploadInput.files.length > 0) {
                    fileLabel.textContent = videoUploadInput.files[0].name;
                    uploadBtn.disabled = false; uploadStatusElement.textContent = '';
                } else {
                    fileLabel.textContent = "Select video file"; uploadBtn.disabled = true;
                }
            });
            uploadBtn.disabled = true;

            async function uploadVideo() {
                console.log("SCRIPT (Upload): uploadVideo called.");
                const file = videoUploadInput.files[0];
                if (!file) { uploadStatusElement.textContent = "Please select a video file first."; uploadStatusElement.style.color = 'red'; return; }

                uploadStatusElement.textContent = "Uploading video..."; uploadStatusElement.style.color = 'var(--primary-color)';
                uploadBtn.disabled = true; fileLabel.style.opacity = '0.6';
                const formData = new FormData(); formData.append('video', file);

                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const result = await response.json();
                    if (response.ok) {
                        uploadStatusElement.textContent = `Upload successful: ${result.message || 'Video received.'}`; uploadStatusElement.style.color = 'green';
                        videoUploadInput.value = ''; fileLabel.textContent = "Select video file"; uploadBtn.disabled = true;
                    } else {
                        uploadStatusElement.textContent = `Upload failed: ${result.error || 'Server error.'}`; uploadStatusElement.style.color = 'red';
                        uploadBtn.disabled = false;
                    }
                } catch (error) {
                    console.error("SCRIPT ERROR (Upload): Fetch error:", error);
                    uploadStatusElement.textContent = "Upload error, please check network."; uploadStatusElement.style.color = 'red';
                    uploadBtn.disabled = false;
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
        console.log("SCRIPT: Herb identifier page container FOUND. Setting up logic...");
        const imageUploadInput = document.getElementById('herbImageUpload');
        const predictBtn = document.getElementById('predictBtn');
        const predictionResultElement = document.getElementById('predictionResult');
        const imagePreview = document.getElementById('imagePreview');
        const fileLabel = herbPageContainer.querySelector('.file-label');

        if (!imageUploadInput || !predictBtn || !predictionResultElement || !imagePreview || !fileLabel) {
            console.error("SCRIPT ERROR: One or more required elements for the herb identifier page are missing!");
        } else {
            console.log("SCRIPT: All herb identifier page elements found.");

            imageUploadInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    if (!file.type.startsWith('image/')){
                         predictionResultElement.textContent = "Please select an image file."; predictionResultElement.style.color = 'red';
                         imagePreview.style.display = 'none'; imagePreview.src = '#';
                         fileLabel.textContent = "Select image file"; predictBtn.disabled = true;
                         return;
                    }
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                    fileLabel.textContent = file.name;
                    predictBtn.disabled = false;
                    predictionResultElement.textContent = ''; predictionResultElement.style.color = 'inherit';
                } else {
                    imagePreview.style.display = 'none'; imagePreview.src = '#';
                    fileLabel.textContent = "Select image file";
                    predictBtn.disabled = true;
                }
            });

            async function predictHerb() {
                const file = imageUploadInput.files[0];
                if (!file) {
                    predictionResultElement.textContent = "Please select an image first."; predictionResultElement.style.color = 'red';
                    return;
                }

                predictionResultElement.textContent = "Identifying, please wait..."; predictionResultElement.style.color = 'var(--primary-color)';
                predictBtn.disabled = true; fileLabel.style.opacity = '0.6';

                const formData = new FormData(); formData.append('image', file);

                try {
                    const response = await fetch('/predict_herb', { method: 'POST', body: formData });
                    const result = await response.json();
                    if (response.ok) {
                        predictionResultElement.innerHTML = `Identification result: <strong>${result.prediction}</strong>${result.confidence ? `<br>(Confidence: ${result.confidence})` : ''}`;
                        predictionResultElement.style.color = 'green';
                    } else {
                        predictionResultElement.textContent = `Identification failed: ${result.error || `Server error (${response.status})`}`;
                        predictionResultElement.style.color = 'red';
                    }
                } catch (error) {
                    console.error("SCRIPT ERROR (Herb): Fetch error:", error);
                    predictionResultElement.textContent = "Identification request error, please check network.";
                    predictionResultElement.style.color = 'red';
                } finally {
                    predictBtn.disabled = false;
                    fileLabel.style.opacity = '1';
                }
            }
            if (predictBtn) {
                 predictBtn.addEventListener('click', predictHerb);
            }
        }
    } else {
        console.log("SCRIPT: Herb identifier page container NOT FOUND. Skipping logic setup.");
    }
    console.log("SCRIPT: End of DOMContentLoaded execution.");
});


// ========================================================
// ==  全局悬浮聊天机器人 (Global Floating Chatbot) - 已修改  ==
// ========================================================
document.addEventListener('DOMContentLoaded', () => {
    const chatbotFab = document.getElementById('chatbot-fab');
    const chatbotContainer = document.getElementById('chatbot-container');
    const closeBtn = document.getElementById('chatbot-close-btn');
    const chatWindow = document.getElementById('chat-window');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    if (!chatbotFab || !chatbotContainer) {
        return;
    }

    // --- [新增 ✨] 创建一个变量来存储聊天记录 ---
    let chatHistory = [];

    // --- 事件监听：控制聊天窗口的显示/隐藏 ---
    chatbotFab.addEventListener('click', () => {
        chatbotContainer.classList.toggle('is-hidden');
        if (!chatbotContainer.classList.contains('is-hidden')) {
            chatInput.focus();
        }
    });

    closeBtn.addEventListener('click', () => {
        chatbotContainer.classList.add('is-hidden');
    });

    // --- 聊天核心逻辑 ---
    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', `${sender}-message`);
        messageElement.textContent = text;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageElement;
    }

    async function handleSendMessage() {
        const userQuestion = chatInput.value.trim();
        if (!userQuestion) return;

        addMessage(userQuestion, 'user');
        chatInput.value = '';
        sendBtn.disabled = true;

        const thinkingMessage = addMessage('...', 'bot');

        try {
            // --- [修改 ✨] 发送请求时，带上当前的聊天记录 ---
            const response = await fetch('/ask_chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: userQuestion,
                    history: chatHistory // 将历史记录发送给后端
                }),
            });

            chatWindow.removeChild(thinkingMessage);

            const result = await response.json();
            if (response.ok) {
                addMessage(result.answer, 'bot');
                // --- [新增 ✨] 用后端返回的最新历史记录，更新前端的变量 ---
                chatHistory = result.history;
            } else {
                addMessage(`Error: ${result.error || 'Unknown Error'}`, 'bot');
            }
        } catch (error) {
            chatWindow.removeChild(thinkingMessage);
            addMessage(`Network request failed, please check if the back-end service is running normally.`, 'bot');
            console.error('Fetch error:', error);
        } finally {
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    });

    // 初始欢迎语
    addMessage('Hello! I am your dedicated TCM health assistant, how can I help you?', 'bot');
});
