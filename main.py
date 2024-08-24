import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
import requests
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)

# Initialize SAM 2 model
checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

@app.route('/')
def hello_world():
    cuda_available = torch.cuda.is_available()
    return f'Hello, World! CUDA is {"available" if cuda_available else "not available"}.'

@app.route('/segment', methods=['POST'])
def segment_image():
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    # Download the image
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # Perform segmentation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(point_coords=[[image.width // 2, image.height // 2]])

    # Convert mask to binary list for JSON serialization
    mask_list = masks[0].cpu().numpy().tolist()

    return jsonify({'mask': mask_list})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)