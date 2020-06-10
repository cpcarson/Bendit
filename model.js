const URL = './model_data/';
let model, webcam, ctx, labelContainer, maxPredictions;

async function init() {
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';

    // load model
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

 
    webcam = new tmPose.Webcam(900, 900, flip=true); 
    await webcam.setup(); 
    webcam.play();
    window.requestAnimationFrame(loop);

  
    const canvas = document.getElementById('canvas');
    canvas.width = 900; canvas.height = 720;
    ctx = canvas.getContext('2d');
    labelContainer = document.getElementById('label-container');
    for (let i = 0; i < maxPredictions; i++) { //class labels
        labelContainer.appendChild(document.createElement('div'));
    }
}

async function loop(timestamp) {
    webcam.update();
    await predict();
    window.requestAnimationFrame(loop);
}
async function stop() {
    webcam.stop();
}

async function predict() {

    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas); //output pose

    const prediction = await model.predictTopK(posenetOutput, maxPredictions = 1); //predict pose

    

    for (let i = 0; i < maxPredictions; i++) {
        var classScore = prediction[i].probability.toFixed(1);
        const noScore = "No Pose"    
        const classPrediction =
            prediction[i].className + ': ' + classScore;
        // IF STATEMENT GOES HERE FOR CHANGING COLOR
        if (classScore < 0.8) {
            labelContainer.childNodes[i].innerHTML = noScore;
            document.getElementById("label-container").style.color = "red"; 

        } else {
            labelContainer.childNodes[i].innerHTML = classPrediction;
            document.getElementById("label-container").style.color = "green";
        }
    }

    // draw the poses
    drawPose(pose);
}

function drawPose(pose) {
    ctx.drawImage(webcam.canvas, 0, 0);
    if (pose) {
        const minPartConfidence = 0.5;
        tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
        tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
    }
}

