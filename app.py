from flask import Flask, render_template, request, redirect, url_for, flash
import os
import subprocess
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define the path to your model repository
REPO_DIR = "model"
WEIGHT_DIR = os.path.join(REPO_DIR, "weight")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            video_path = os.path.join(REPO_DIR, file.filename)
            file.save(video_path)

            # Define the command to run prediction
            prediction_command = (
                f"python {os.path.join(REPO_DIR, 'pred_func.py')} "
                f"--p {video_path} "
                f"--f {10} "  # dynamically pass the number of frames
                f"--e genconvit_ed --v genconvit_vae"
            )

            # Execute the prediction command
            try:
                subprocess.run(prediction_command, shell=True, check=True)
                result_path = os.path.join(REPO_DIR, "result", "result_all.txt")
                if Path(result_path).exists():
                    with open(result_path, 'r') as result_file:
                        results = result_file.read()
                    flash(f"Prediction results: {results}")
                else:
                    flash("Result file not found")
            except subprocess.CalledProcessError as e:
                flash(f"Error running the model: {e}")

            return redirect(url_for("index"))
    return render_template("index.html")


@app.route("/download_weights")
def download_weights():
    # Download weights logic
    subprocess.run(
        f"wget -P {WEIGHT_DIR} https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth",
        shell=True
    )
    subprocess.run(
        f"wget -P {WEIGHT_DIR} https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth",
        shell=True
    )
    flash("Weights downloaded successfully!")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
