
let currentResolution = { width: 640, height: 480 };  // 기본 해상도 설정


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
async function generateAnswer() {
    const clueInput = document.getElementById("clue_input");
    const answerOutput = document.getElementById("answer_output");
    const latencyOutput = document.getElementById("latency_output");
    const objectList = document.getElementById("object_list");

    const clue = clueInput.value.trim();

    if (!clue) {
        alert("Enter your clue!");
        return;
    }

    document.getElementById("status").innerText = "Generating answers with LLM...";
    
    try {
        const response = await fetch("/generate_answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clue }),
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
        objectList.innerHTML = ""; // 기존 리스트 초기화
        console.log(data.positions);

        if (!data.positions) {
            // 매칭된 객체가 없을 때
            const li = document.createElement("li");
            li.textContent = "No matching object found.";
            objectList.appendChild(li);
        } else {
            // 매칭된 객체가 있을 때
            const li = document.createElement("li");
            li.innerHTML = `
                <strong>Object:</strong> ${data.obj_data}<br>
                <strong>Coordinates:</strong> 
                <ul>
                    <li><strong>X1:</strong> ${data.x1}</li>
                    <li><strong>Y1:</strong> ${data.y1}</li>
                    <li><strong>X2:</strong> ${data.x2}</li>
                    <li><strong>Y2:</strong> ${data.y2}</li>
                </ul>
            `;
            objectList.appendChild(li);
        }
    } catch (err) {
        console.error("Error generating answer:", err);
        document.getElementById("status").innerText = "Error generating answer!";
    }
}
