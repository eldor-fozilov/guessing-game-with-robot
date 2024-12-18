
let currentResolution = { width: 560, height: 400 };  // 기본 해상도 설정
let rejectedObjects = [];
let rejectedObject = null;
let wrongAnswer = false;

fetchUseRobotlStatus();

async function fetchUseRobotlStatus() {
    try {
        const response = await fetch("/get_use_robot_status");
        const data = await response.json();
        useRobot = data.use_robot;
        console.log(`Robot Use: ${useRobot ? "enabled" : "disabled"}`);
    } catch (err) {
        console.error("Error fetching Robot Use status:", err);
    }
}

async function searchCameras() {
    try {
        const response = await fetch('/start_camera_search', { method: 'POST' });
        if (response.ok) {
            document.getElementById("status").innerText = "Searching for cameras...";
            checkCameraResults();
        } else {
            document.getElementById("status").innerText = "Camera search failed!";
        }
    } catch (err) {
        console.error("Error during camera search:", err);
        document.getElementById("status").innerText = "Error searching for cameras!";
    }
}


async function checkCameraResults() {
    try {
        const response = await fetch('/get_camera_results');
        const data = await response.json();
        if (data.status === "completed") {
            const select = document.getElementById("camera_select");
            select.innerHTML = ""; // 기존 옵션 제거
            data.cameras.forEach(camera => {
                const option = document.createElement("option");
                option.value = camera;
                option.text = `Camera ${camera}`;
                select.appendChild(option);
            });
            document.getElementById("status").innerText = "Camera search complete!";
        } else {
            setTimeout(checkCameraResults, 1000); // 1초 후 다시 확인
        }
    } catch (err) {
        console.error("Error fetching camera results:", err);
        document.getElementById("status").innerText = "Error checking camera results!";
    }
}

async function connectCamera() {
    const select = document.getElementById("camera_select");
    const cameraIndex = select.value;

    if (!cameraIndex) {
        alert("카메라를 선택하세요!");
        return;
    }

    try {
        document.getElementById("status").innerText = "Connecting camera...";
        const response = await fetch('/connect_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_index: cameraIndex })
        });
        const data = await response.json();

        if (data.status === "connected") {
            document.getElementById("status").innerText = "Camera connected successfully!";
            // 카메라 스트림 표시
            const videoFeed = document.getElementById("video_feed");
            videoFeed.src = `/video_feed?width=${currentResolution.width}&height=${currentResolution.height}`;
            videoFeed.style.display = "block";
        } else {
            document.getElementById("status").innerText = `Error: ${data.message}`;
        }
    } catch (err) {
        console.error("Error connecting to camera:", err);
        document.getElementById("status").innerText = "Error connecting camera!";
    }
}

// When push the "Generate Answer Button"
async function generateAnswer() {
    const clueInput = document.getElementById("clue_input");
    const answerOutput = document.getElementById("answer_output");
    const latencyOutput = document.getElementById("latency_output");
    const objectList = document.getElementById("object_list");

    // reset wrongAnswer to False
    wrongAnswer = false;

    // send the wrongAnswer to the server
    await fetch("/wrong_answer_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ wrongAnswer }),
        });

    const clue = clueInput.value.trim();

    if (!clue) {
        alert("Enter your clue!");
        return;
    }

    document.getElementById("status").innerText = "Generating answer ...";
    
    try {
        const response = await fetch("/generate_answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clue, rejectedObjects}),
        });

        if (!response.ok) {
            const errorData = await response.json();
            document.getElementById("status").innerText = `Error: ${errorData.error}`;
            return;
        }

        const data = await response.json();

        // 출력 결과 업데이트
        answerOutput.textContent = `Answer: ${data.answer}`;
        latencyOutput.textContent = `Latency: ${data.latency}`;
        document.getElementById("status").innerText = "Answer created!";

        // 감지된 객체 및 좌표 표시
        console.log(data.positions);

        if (!data.positions) {
            document.getElementById("point0").innerText = "No matching object found.";
            objectList.appendChild(li);
        } else {
            // 매칭된 객체가 있을 때
            document.getElementById("point0").innerText = "Point 0 {x1, y1} : {" + data.x1 + ", " + data.y1 + "}";
            document.getElementById("point1").innerText = "Point 1 {x2, y2} : {" + data.x2 + ", " + data.y2 + "}";
        }

        if (useRobot){

        // Contol Robot ----------------------------------------------------------
        const response2 = await fetch("/control_robot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ point0, point1 }),
        });
        const data2 = await response2.json();
        if (response2.ok) {
            document.getElementById("status").innerText = "Robot is moving!";
        } else {
            document.getElementById("status").innerText = `Error: ${data2.error}`;
        }
        // ------------------------------------------------------------------------
    }

    } catch (err) {
        console.error("Error generating answer:", err);
        document.getElementById("status").innerText = "Error generating answer!";
    }
}

async function selectSolution() {
    const select = document.getElementById("solution_select");
    const solution = select.value;

    if (!solution) {
        alert("Select a solution!");
        return;
    }

    try {
        document.getElementById("status").innerText = "Connecting Solution ....";
        
        const response = await fetch("/select_solution", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ solution }),
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById("status").innerText = `Solution selected: ${data.current_solution}`;
        } else {
            document.getElementById("status").innerText = `Error: ${data.error}`;
        }
    } catch (err) {
        console.error("Error selecting solution:", err);
        document.getElementById("status").innerText = "Error selecting solution!";
    }
}

async function startRecording() {
    document.getElementById("status").innerText = "Starting recording...";
    try {
        const response = await fetch("/start_listen", { method: "POST" });
        if (response.ok) {
            document.getElementById("status").innerText = "Recording audio...";
            // Show recording indicator
            document.getElementById("recording_indicator").style.display = "inline";
        } else {
            const errorData = await response.json();
            document.getElementById("status").innerText = "Error starting recording: " + errorData.status;
        }
    } catch (err) {
        console.error("Error starting recording:", err);
        document.getElementById("status").innerText = "Error starting recording!";
    }
}

async function stopRecording() {
    document.getElementById("status").innerText = "Stopping recording...";
    try {
        const response = await fetch("/stop_listen", { method: "POST" });
        const data = await response.json();
        if (response.ok) {
            document.getElementById("status").innerText = "Recording stopped and processed!";
            // Hide recording indicator
            document.getElementById("recording_indicator").style.display = "none";

            // Set the transcription into the clue input textbox
            if (data.transcription) {
                const clueInput = document.getElementById("clue_input");
                clueInput.value = data.transcription;
            }
        } else {
            document.getElementById("status").innerText = "Error stopping recording: " + data.error;
        }
    } catch (err) {
        console.error("Error stopping recording:", err);
        document.getElementById("status").innerText = "Error stopping recording!";
    }
}

async function playLLMOutput() {
    document.getElementById("status").innerText = "Playing output as speech...";
    try {
        const response = await fetch("/speak_llm_output", {
            method: "POST"
        });
        if (response.ok) {
            document.getElementById("status").innerText = "Speech played successfully!";
        } else {
            const errorData = await response.json();
            document.getElementById("status").innerText = "Error playing speech: " + errorData.error;
        }
    } catch (err) {
        console.error("Error playing speech:", err);
        document.getElementById("status").innerText = "Error playing speech!";
    }
}

async function acceptDecision() {
    document.getElementById("status").innerText = "Decision accepted! 🎉";

    const confettiCanvas = document.createElement("canvas");
    confettiCanvas.id = "confetti-canvas";
    confettiCanvas.width = window.innerWidth;
    confettiCanvas.height = window.innerHeight;
    confettiCanvas.style.position = "fixed";
    confettiCanvas.style.top = "0";
    confettiCanvas.style.left = "0";
    confettiCanvas.style.pointerEvents = "none";
    confettiCanvas.style.zIndex = "1000";
    document.body.appendChild(confettiCanvas);

    // reset wrongAnswer to False
    wrongAnswer = false;
    // resert excludedObjects to empty
    rejectedObjects = [];
    // reset rejectedObject to null
    rejectedObject = null;

    // send the wrongAnswer to the server
    await fetch("/wrong_answer_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ wrongAnswer }),
        });

    const ctx = confettiCanvas.getContext("2d");

    const particles = [];
    const colors = ["#ff6347", "#ffa500", "#ffff00", "#32cd32", "#1e90ff", "#ff1493", "#9400d3", "#00ced1"];

    for (let i = 0; i < 200; i++) {
        particles.push({
            x: Math.random() * confettiCanvas.width,
            y: Math.random() * confettiCanvas.height - confettiCanvas.height,
            size: Math.random() * 10 + 5,
            color: colors[Math.floor(Math.random() * colors.length)],
            speedX: Math.random() * 4 - 2,
            speedY: Math.random() * 4 + 2,
            rotation: Math.random() * 360,
            rotationSpeed: Math.random() * 10 - 5,
        });
    }

    function animate() {
        ctx.clearRect(0, 0, confettiCanvas.width, confettiCanvas.height);
        particles.forEach((p) => {
            p.x += p.speedX;
            p.y += p.speedY;
            p.rotation += p.rotationSpeed;

            if (p.y > confettiCanvas.height) {
                p.y = -p.size;
                p.x = Math.random() * confettiCanvas.width;
            }

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate((p.rotation * Math.PI) / 180);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
            ctx.restore();
        });
        requestAnimationFrame(animate);
    }

    animate();

    setTimeout(() => {
        document.body.removeChild(confettiCanvas);
        document.getElementById("status").innerText = "Ready for the next task!";
    }, 5000);
}

async function rejectDecision() {
    const answerOutput = document.getElementById("answer_output");
    rejectedObject = answerOutput.textContent.split(": ")[1]?.trim();
    console.log("rejectedObject:", rejectedObject);
    // Set wrongAnswer to true
    wrongAnswer = true;

    // Send the wrongAnswer status to the server
    await fetch("/wrong_answer_status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wrongAnswer }),
    });

    if (!rejectedObject) {
        alert("No object to reject. Generate an answer first!");
        return;
    }

    // Add the rejected object to the array if it's not already there
    if (!rejectedObjects.includes(rejectedObject)) {
        rejectedObjects.push(rejectedObject);
    }

    console.log("Rejected Objects Array:", rejectedObjects);

    alert("The answer was rejected. You can record / enter a new clue or click the 'Generate Answer' button to regenerate the answer.");
}